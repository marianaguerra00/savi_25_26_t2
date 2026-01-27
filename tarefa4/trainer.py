import torch
import torch.nn as nn
from torch import optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import time
import random
import torch.nn.functional as F


def calculateIoU(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two boxes.
    
    Args:
        box1, box2: (x, y, w, h) format
    
    Returns:
        iou: float between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    xLeft = max(x1, x2)
    yTop = max(y1, y2)
    xRight = min(x1 + w1, x2 + w2)
    yBottom = min(y1 + h1, y2 + h2)
    
    if xRight < xLeft or yBottom < yTop:
        return 0.0
    
    intersection = (xRight - xLeft) * (yBottom - yTop)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


class ImprovedTrainer:
    """
    FPN-based trainer with multi-scale loss and proper bbox handling.
    
    Key features:
    - Multi-scale training (P3: 32x32, P4: 16x16)
    - Weighted losses (conf, class, bbox)
    - Detection metrics (IoU, detection accuracy)
    - Learning rate scheduling
    """
    def __init__(self, model, trainDataset, testDataset, args):
        self.model = model
        self.args = args

        self.trainLossHistory = []
        self.testLossHistory = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Loss functions
        self.confLoss = nn.BCEWithLogitsLoss()
        self.classLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.bboxLoss = nn.MSELoss()
        
        # Loss weights
        self.confWeight = 2.0   # Higher weight for objectness
        self.classWeight = 1.5  # Higher weight for classification
        self.bboxWeight = 1.0   # Standard weight for bbox

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args["learningRate"],
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

        self.trainLoader = torch.utils.data.DataLoader(
            trainDataset,
            batch_size=args["batchSize"],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        self.testLoader = torch.utils.data.DataLoader(
            testDataset,
            batch_size=args["batchSize"],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Grid info for bbox decoding
        self.imageSize = 128
        self.gridSize = 32
        self.cellSize = self.imageSize / self.gridSize

    def computeLossSingleScale(self, outputs, confTarget, classTarget, bboxTarget):
        """
        Compute loss at a single scale.
        
        Args:
            outputs: [B, 15, H, W] model predictions
            confTarget: [B, H, W] confidence targets
            classTarget: [B, H, W] class targets
            bboxTarget: [B, 4, H, W] bbox targets
        
        Returns:
            totalLoss: weighted sum of conf + class + bbox losses
        """
        # Split predictions
        confPred = outputs[:, 0]           # [B, H, W]
        classPred = outputs[:, 1:11]       # [B, 10, H, W]
        bboxPred = outputs[:, 11:15]       # [B, 4, H, W] - already processed by model

        # Confidence loss
        lossConf = self.confLoss(confPred, confTarget) * self.confWeight

        # Classification loss (only for cells with objects)
        lossClass = self.classLoss(
            classPred.permute(0, 2, 3, 1).reshape(-1, 10),
            classTarget.view(-1)
        ) * self.classWeight

        # Bbox loss (only for cells with objects)
        objectMask = confTarget > 0.5
        if objectMask.sum() > 0:
            bboxPredPerm = bboxPred.permute(0, 2, 3, 1)
            bboxTargetPerm = bboxTarget.permute(0, 2, 3, 1)
            lossBbox = self.bboxLoss(
                bboxPredPerm[objectMask],
                bboxTargetPerm[objectMask]
            ) * self.bboxWeight
        else:
            lossBbox = torch.tensor(0.0, device=self.device)

        return lossConf + lossClass + lossBbox

    def trainEpoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        losses = []

        for images, confTarget, classTarget, bboxTarget in tqdm(
            self.trainLoader,
            desc=f"Train Epoch {epoch + 1}",
            leave=False
        ):
            # Move data to device
            images = images.to(self.device)
            confTarget = confTarget.to(self.device)
            classTarget = classTarget.to(self.device)
            bboxTarget = bboxTarget.to(self.device)

            # Forward pass (FPN outputs)
            outP3, outP4 = self.model(images)

            # Compute loss at main scale (P3: 32x32)
            lossP3 = self.computeLossSingleScale(
                outP3, confTarget, classTarget, bboxTarget
            )

            # Downsample targets for P4 (16x16)
            confTargetP4 = F.max_pool2d(confTarget, 2)
            classTargetP4 = F.max_pool2d(classTarget.float(), 2).long()
            bboxTargetP4 = F.max_pool2d(bboxTarget, 2)

            # Compute loss at secondary scale (P4: 16x16)
            lossP4 = self.computeLossSingleScale(
                outP4, confTargetP4, classTargetP4, bboxTargetP4
            )

            # Combined weighted loss
            loss = lossP3 + 0.7 * lossP4

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            losses.append(loss.item())

        return np.mean(losses)

    def evaluate(self, epoch):
        """Evaluate on validation set"""
        self.model.eval()
        losses = []

        with torch.no_grad():
            for images, confTarget, classTarget, bboxTarget in tqdm(
                self.testLoader,
                desc=f"Validation Epoch {epoch + 1}",
                leave=False
            ):
                images = images.to(self.device)
                confTarget = confTarget.to(self.device)
                classTarget = classTarget.to(self.device)
                bboxTarget = bboxTarget.to(self.device)

                # Forward pass
                outP3, outP4 = self.model(images)

                # Multi-scale loss
                lossP3 = self.computeLossSingleScale(
                    outP3, confTarget, classTarget, bboxTarget
                )
                lossP4 = self.computeLossSingleScale(
                    outP4,
                    F.max_pool2d(confTarget, 2),
                    F.max_pool2d(classTarget.float(), 2).long(),
                    F.max_pool2d(bboxTarget, 2)
                )

                loss = lossP3 + 0.7 * lossP4
                losses.append(loss.item())

        return np.mean(losses)

    def decodeBBox(self, gridY, gridX, tx, ty, tw, th):
        """
        Decode bbox from grid cell offsets to absolute coordinates.
        
        Args:
            gridY, gridX: grid cell indices
            tx, ty: center offsets within cell (can be any value)
            tw, th: width/height normalized by image size (0-1, already sigmoid'd)
        
        Returns:
            x, y, w, h: absolute pixel coordinates
        """
        # Calculate absolute center
        cellLeft = gridX * self.cellSize
        cellTop = gridY * self.cellSize
        
        cx = cellLeft + tx * self.cellSize
        cy = cellTop + ty * self.cellSize
        
        # Calculate absolute width/height
        w = tw * self.imageSize
        h = th * self.imageSize
        
        # Convert to top-left corner
        x = cx - w / 2
        y = cy - h / 2
        
        return x, y, w, h

    def evaluateFinal(self, maxImagesToShow=50, confThreshold=0.5):
        """
        Final evaluation with metrics and visualization.
        
        Args:
            maxImagesToShow: number of images to show in UI
            confThreshold: confidence threshold for considering a detection
        """
        self.model.eval()

        allResults = []
        allTrueLabels = []
        allPredLabels = []
        allIoUs = []
        detectionCorrect = 0
        detectionTotal = 0

        startTime = time.time()

        with torch.no_grad():
            for images, confTarget, classTarget, bboxTarget in tqdm(
                self.testLoader, desc="Final Evaluation"
            ):
                images = images.to(self.device)
                confTarget = confTarget.to(self.device)
                classTarget = classTarget.to(self.device)
                bboxTarget = bboxTarget.to(self.device)

                # Forward pass - use P3 scale for evaluation
                outP3, _ = self.model(images)

                confPred = torch.sigmoid(outP3[:, 0])    # [B, H, W]
                classPred = outP3[:, 1:11]               # [B, 10, H, W]
                bboxPred = outP3[:, 11:15]               # [B, 4, H, W]

                B = confPred.shape[0]

                for b in range(B):
                    # Only evaluate cells with ground truth objects
                    objectMask = confTarget[b] > confThreshold
                    if objectMask.sum() == 0:
                        continue

                    # Classification metrics
                    classPredFlat = classPred[b].permute(1, 2, 0)[objectMask]
                    classTrueFlat = classTarget[b][objectMask]
                    predictedLabels = torch.argmax(classPredFlat, dim=1)

                    allTrueLabels.extend(classTrueFlat.cpu().numpy())
                    allPredLabels.extend(predictedLabels.cpu().numpy())

                    # Visualization + detection metrics
                    boxes = []
                    labels = []
                    gtLabels = []
                    confidences = []
                    gtBoxes = []

                    bboxFlat = bboxPred[b].permute(1, 2, 0)
                    bboxTrueFlat = bboxTarget[b].permute(1, 2, 0)
                    confFlat = confPred[b]
                    ys, xs = objectMask.nonzero(as_tuple=True)

                    for (cy, cx), (tx, ty, tw, th), (txGt, tyGt, twGt, thGt), lbl, gtLbl in zip(
                        zip(ys.cpu().numpy(), xs.cpu().numpy()),
                        bboxFlat[objectMask].cpu().numpy(),
                        bboxTrueFlat[objectMask].cpu().numpy(),
                        predictedLabels.cpu().numpy(),
                        classTrueFlat.cpu().numpy()
                    ):
                        # Decode predicted bbox
                        xPred, yPred, wPred, hPred = self.decodeBBox(
                            cy, cx, tx, ty, tw, th
                        )

                        # Decode ground truth bbox
                        xGt, yGt, wGt, hGt = self.decodeBBox(
                            cy, cx, txGt, tyGt, twGt, thGt
                        )

                        # Calculate IoU
                        iou = calculateIoU(
                            (xPred, yPred, wPred, hPred),
                            (xGt, yGt, wGt, hGt)
                        )

                        allIoUs.append(iou)
                        detectionTotal += 1
                        if iou > 0.5:
                            detectionCorrect += 1

                        boxes.append((float(xPred), float(yPred), float(wPred), float(hPred)))
                        gtBoxes.append((float(xGt), float(yGt), float(wGt), float(hGt)))
                        labels.append(int(lbl))
                        gtLabels.append(int(gtLbl))
                        confidences.append(float(confFlat[cy, cx]))

                    allResults.append({
                        "image": images[b].cpu().squeeze().numpy(),
                        "boxes": boxes,
                        "gtBoxes": gtBoxes,
                        "labels": labels,
                        "gtLabels": gtLabels,
                        "confidences": confidences
                    })

        # Shuffle and limit results for visualization
        random.shuffle(allResults)
        results = allResults[:maxImagesToShow]

        totalTime = time.time() - startTime

        # Calculate metrics
        accuracy = accuracy_score(allTrueLabels, allPredLabels)
        precision = precision_score(allTrueLabels, allPredLabels, average="macro", zero_division=0)
        recall = recall_score(allTrueLabels, allPredLabels, average="macro", zero_division=0)
        f1 = f1_score(allTrueLabels, allPredLabels, average="macro", zero_division=0)
        cm = confusion_matrix(allTrueLabels, allPredLabels)

        meanIoU = np.mean(allIoUs) if allIoUs else 0.0
        detectionPrecision = (
            detectionCorrect / detectionTotal if detectionTotal > 0 else 0.0
        )

        # Print results
        print("\n" + "=" * 50)
        print("FINAL TEST RESULTS")
        print("=" * 50)
        print("Classification Metrics:")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1-score : {f1:.4f}")
        print("\nDetection Metrics:")
        print(f"  Mean IoU        : {meanIoU:.4f}")
        print(f"  Detection Acc   : {detectionPrecision:.4f} (IoU > 0.5)")
        print(f"  Total detections: {detectionTotal}")
        print(f"\nEvaluation time: {totalTime:.2f} s")
        print("=" * 50)

        if len(results) == 0:
            print("No images available for visualization.")
            return

        self.showEvaluationUI(
            accuracy, results, precision, recall, f1, totalTime, cm,
            meanIoU, detectionPrecision
        )

    def showEvaluationUI(self, accuracy, results, precision, recall, f1Score, 
                        totalTime, cm, meanIoU, detectionPrecision):
        """Interactive visualization UI"""
        currentImageIndex = [0]
        currentStatsIndex = [0]

        fig = plt.figure(figsize=(10, 7))

        axImage = fig.add_axes([0.15, 0.50, 0.45, 0.40])
        axStats = fig.add_axes([0.08, 0.10, 0.60, 0.30])

        axNextImg = fig.add_axes([0.72, 0.65, 0.25, 0.10])
        axPrevImg = fig.add_axes([0.72, 0.53, 0.25, 0.10])
        axNextStat = fig.add_axes([0.72, 0.25, 0.25, 0.10])
        axPrevStat = fig.add_axes([0.72, 0.13, 0.25, 0.10])

        fig.text(
            0.98, 0.98,
            f"Processing time: {totalTime:.2f}s",
            ha="right", va="top",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.7),
            color="white"
        )

        def drawImage():
            axImage.clear()
            data = results[currentImageIndex[0]]

            axImage.imshow(data["image"], cmap="gray")
            axImage.axis("off")

            # Draw ground truth boxes (blue dashed)
            for (xGt, yGt, wGt, hGt) in data["gtBoxes"]:
                axImage.add_patch(
                    patches.Rectangle(
                        (xGt, yGt), wGt, hGt,
                        linewidth=1.5,
                        edgecolor="blue",
                        facecolor="none",
                        linestyle="--"
                    )
                )

            # Draw predicted boxes
            for (x, y, w, h), label, gtLabel, conf, (xGt, yGt, wGt, hGt) in zip(
                data["boxes"],
                data["labels"],
                data["gtLabels"],
                data["confidences"],
                data["gtBoxes"]
            ):
                iou = calculateIoU((x, y, w, h), (xGt, yGt, wGt, hGt))
                
                isCorrect = (label == gtLabel)
                color = "lime" if isCorrect else "red"

                axImage.add_patch(
                    patches.Rectangle(
                        (x, y), w, h,
                        linewidth=2,
                        edgecolor=color,
                        facecolor="none"
                    )
                )

                axImage.text(
                    x + 2,
                    y - 6 if y > 10 else y + h + 6,
                    f"{label} ({conf:.2f}) IoU:{iou:.2f}",
                    color="white",
                    fontsize=6,
                    bbox=dict(facecolor=color, alpha=0.85)
                )

            axImage.set_title(
                f"Image {currentImageIndex[0] + 1}/{len(results)}\n"
                f"Blue Dash=GT | Green=Correct | Red=Wrong"
            )
            fig.canvas.draw_idle()

        def drawDetectionMetrics():
            """Index 0: Detection metrics"""
            axStats.clear()
            
            metrics = [meanIoU, detectionPrecision]
            labels = [
                f"Mean IoU\n{meanIoU:.3f}",
                f"Detection Acc\n{detectionPrecision:.3f}\n(IoU > 0.5)"
            ]
            colors = ['#3498db', '#2ecc71']
            
            bars = axStats.bar(labels, metrics, color=colors, alpha=0.7, width=0.5)
            axStats.set_ylim(0, 1)
            axStats.set_title("Detection Metrics (BBox Quality)", fontweight='bold')
            axStats.grid(True, alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                axStats.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            axStats.text(0.5, -0.15, 
                        "IoU measures bbox overlap | Detection Acc = % boxes with IoU > 0.5",
                        ha='center', va='top', transform=axStats.transAxes,
                        fontsize=8, style='italic', color='gray')

        def drawClassificationMetrics():
            """Index 1: Classification metrics"""
            axStats.clear()
            values = [accuracy, precision, recall, f1Score]
            labels = [
                f"Accuracy\n{accuracy:.3f}",
                f"Precision\n{precision:.3f}",
                f"Recall\n{recall:.3f}",
                f"F1\n{f1Score:.3f}"
            ]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            bars = axStats.bar(labels, values, color=colors, alpha=0.7)
            axStats.set_ylim(0, 1)
            axStats.set_title("Classification Metrics (Digit Recognition)", fontweight='bold')
            axStats.grid(True, alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                axStats.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')

        def drawConfusionMatrix():
            """Index 2: Confusion matrix"""
            axStats.clear()
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axStats,
                cbar=True,
                square=True
            )
            axStats.set_title("Confusion Matrix (Digit Classes)", fontweight='bold')
            axStats.set_xlabel("Predicted")
            axStats.set_ylabel("True")

        statsFns = [drawDetectionMetrics, drawClassificationMetrics, drawConfusionMatrix]

        def drawStats():
            statsFns[currentStatsIndex[0]]()
            fig.canvas.draw_idle()

        def onNextImage(event):
            currentImageIndex[0] = min(len(results) - 1, currentImageIndex[0] + 1)
            drawImage()

        def onPrevImage(event):
            currentImageIndex[0] = max(0, currentImageIndex[0] - 1)
            drawImage()

        def onNextStats(event):
            currentStatsIndex[0] = (currentStatsIndex[0] + 1) % len(statsFns)
            drawStats()

        def onPrevStats(event):
            currentStatsIndex[0] = (currentStatsIndex[0] - 1) % len(statsFns)
            drawStats()

        btnNextImg = Button(axNextImg, "Next →")
        btnPrevImg = Button(axPrevImg, "← Previous")
        btnNextStat = Button(axNextStat, "Next Stats →")
        btnPrevStat = Button(axPrevStat, "← Previous Stats")

        btnNextImg.on_clicked(onNextImage)
        btnPrevImg.on_clicked(onPrevImage)
        btnNextStat.on_clicked(onNextStats)
        btnPrevStat.on_clicked(onPrevStats)

        drawImage()
        drawStats()
        plt.show()

    def plotLossCurves(self):
        """Plot and save loss curves"""
        epochs = range(1, len(self.trainLossHistory) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.trainLossHistory, "b-o", label="Train Loss", 
                linewidth=2, markersize=5)
        plt.plot(epochs, self.testLossHistory, "r-s", label="Val Loss", 
                linewidth=2, markersize=5)

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training and Validation Loss", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("loss_vs_epochs.png", dpi=150)
        print("Loss curve saved as 'loss_vs_epochs.png'")
        plt.close()

    def train(self):
        """Main training loop"""
        print(f"\nStarting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        bestLoss = float('inf')
        
        for epoch in range(self.args["numEpochs"]):
            print(f"\nEpoch [{epoch + 1}/{self.args['numEpochs']}]")
            
            trainLoss = self.trainEpoch(epoch)
            testLoss = self.evaluate(epoch)

            self.trainLossHistory.append(trainLoss)
            self.testLossHistory.append(testLoss)
            
            # Update learning rate
            self.scheduler.step(testLoss)

            print(f"Train Loss: {trainLoss:.4f} | Val Loss: {testLoss:.4f}")
            
            # Save best model
            if testLoss < bestLoss:
                bestLoss = testLoss
                torch.save(self.model.state_dict(), "best_model.pth")
                print(f"✓ Best model saved (loss: {bestLoss:.4f})")

        print("\n" + "="*50)
        print("Training finished!")
        print("="*50)
        
        self.plotLossCurves()
        self.evaluateFinal()

        torch.save(self.model.state_dict(), "final_model.pth")
        print("\nFinal model saved as 'final_model.pth'")