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
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import time
import random



class Trainer:
    def __init__(self, model, trainDataset, testDataset, args):
        self.model = model
        self.args = args

        self.trainLossHistory = []
        self.testLossHistory = []



        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.confLoss = nn.BCEWithLogitsLoss()
        self.classLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.bboxLoss = nn.MSELoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args["learningRate"]
        )

        self.trainLoader = torch.utils.data.DataLoader(
            trainDataset,
            batch_size=args["batchSize"],
            shuffle=True
        )

        self.testLoader = torch.utils.data.DataLoader(
            testDataset,
            batch_size=args["batchSize"],
            shuffle=False
        )

    def trainEpoch(self, epoch):
        self.model.train()
        losses = []

        for images, confT, classT, bboxT in tqdm(
            self.trainLoader,
            desc=f"Train Epoch {epoch + 1}",
            leave=False
        ):
            images = images.to(self.device)
            confT = confT.to(self.device)
            classT = classT.to(self.device)
            bboxT = bboxT.to(self.device)

            outputs = self.model(images)

            confPred = outputs[:, 0]
            classPred = outputs[:, 1:11]
            bboxPred = outputs[:, 11:15]

            lossConf = self.confLoss(confPred, confT)

            lossClass = self.classLoss(
                classPred.permute(0, 2, 3, 1).reshape(-1, 10),
                classT.view(-1)
            )

            objectMask = confT > 0.5
            if objectMask.sum() > 0:
                bboxPredPerm = bboxPred.permute(0, 2, 3, 1)
                bboxTPerm = bboxT.permute(0, 2, 3, 1)
                lossBBox = self.bboxLoss(
                    bboxPredPerm[objectMask],
                    bboxTPerm[objectMask]
                )
            else:
                lossBBox = torch.tensor(0.0, device=self.device)

            loss = lossConf + lossClass + lossBBox

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        return np.mean(losses)

    def evaluate(self, epoch):
        self.model.eval()
        losses = []

        with torch.no_grad():
            for images, confT, classT, bboxT in tqdm(
                self.testLoader,
                desc=f"Validation Epoch {epoch + 1}",
                leave=False
            ):
                images = images.to(self.device)
                confT = confT.to(self.device)
                classT = classT.to(self.device)
                bboxT = bboxT.to(self.device)

                outputs = self.model(images)

                confPred = outputs[:, 0]
                classPred = outputs[:, 1:11]
                bboxPred = outputs[:, 11:15]

                lossConf = self.confLoss(confPred, confT)

                lossClass = self.classLoss(
                    classPred.permute(0, 2, 3, 1).reshape(-1, 10),
                    classT.view(-1)
                )

                objectMask = confT > 0.5
                if objectMask.sum() > 0:
                    bboxPredPerm = bboxPred.permute(0, 2, 3, 1)
                    bboxTPerm = bboxT.permute(0, 2, 3, 1)
                    lossBBox = self.bboxLoss(
                        bboxPredPerm[objectMask],
                        bboxTPerm[objectMask]
                    )
                else:
                    lossBBox = torch.tensor(0.0, device=self.device)

                loss = lossConf + lossClass + lossBBox

                losses.append(loss.item())

        return np.mean(losses)
    
    import time

    def evaluateFinal(self, maxImagesToShow=50):
        self.model.eval()


        allResults = []

        allTrueLabels = []
        allPredLabels = []
        results = []

        startTime = time.time()

        with torch.no_grad():
            for images, confT, classT, bboxT in self.testLoader:
                images = images.to(self.device)
                confT = confT.to(self.device)
                classT = classT.to(self.device)

                outputs = self.model(images)

                confPred = torch.sigmoid(outputs[:, 0])   # [B, Gh, Gw]
                classPred = outputs[:, 1:11]              # [B, 10, Gh, Gw]
                bboxPred = outputs[:, 11:15]              # [B, 4, Gh, Gw]

                B, Gh, Gw = confPred.shape
                _, _, H, W = images.shape

                cellW = W / Gw
                cellH = H / Gh

                for b in range(B):

                    objectMask = confT[b] > 0.5
                    if objectMask.sum() == 0:
                        continue

                    # -------- MÉTRICAS --------
                    classPredFlat = classPred[b].permute(1, 2, 0)[objectMask]
                    classTrueFlat = classT[b][objectMask]
                    predictedLabels = torch.argmax(classPredFlat, dim=1)

                    allTrueLabels.extend(classTrueFlat.cpu().numpy())
                    allPredLabels.extend(predictedLabels.cpu().numpy())

                    # -------- UI --------
                    boxes = []
                    labels = []
                    gtLabels = []

                    bboxFlat = bboxPred[b].permute(1, 2, 0)
                    ys, xs = objectMask.nonzero(as_tuple=True)

                    for (cy, cx), (tx, ty, tw, th), lbl, gtLbl in zip(
                        zip(ys.cpu().numpy(), xs.cpu().numpy()),
                        bboxFlat[objectMask].cpu().numpy(),
                        predictedLabels.cpu().numpy(),
                        classTrueFlat.cpu().numpy()
                    ):
                        
                        x_center = (cx + tx) * cellW
                        y_center = (cy + ty) * cellH

                        w_px = tw * W
                        h_px = th * H


                        x = x_center - w_px / 2
                        y = y_center - h_px / 2


                        boxes.append((float(x), float(y), float(w_px), float(h_px)))
                        labels.append(int(lbl))
                        gtLabels.append(int(gtLbl))




                    allResults.append({
                        "image": images[b].cpu().squeeze().numpy(),
                        "boxes": boxes,
                        "labels": labels,
                        "gtLabels": gtLabels
                    })

        random.shuffle(allResults)
        results = allResults[:maxImagesToShow]

        totalTime = time.time() - startTime

        # -----------------------------------------
        accuracy = accuracy_score(allTrueLabels, allPredLabels)
        precision = precision_score(allTrueLabels, allPredLabels, average="macro")
        recall = recall_score(allTrueLabels, allPredLabels, average="macro")
        f1 = f1_score(allTrueLabels, allPredLabels, average="macro")
        cm = confusion_matrix(allTrueLabels, allPredLabels)


        print("\nFINAL TEST RESULTS (10k images)")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"Evaluation time: {totalTime:.2f} s")

        if len(results) == 0:
            print("No images available for visualization.")
            return


        self.showEvaluationUI(accuracy, results, precision, recall, f1, totalTime, cm)

    # -----------------------------
    # Vizualization
    # -----------------------------
    def showEvaluationUI(self, accuracy, results, precision, recall, f1Score, totalTime, cm):

        currentImageIndex = [0]
        currentStatsIndex = [0]

        fig = plt.figure(figsize=(9, 6))

        axImage = fig.add_axes([0.17, 0.50, 0.40, 0.40])
        axStats = fig.add_axes([0.08, 0.10, 0.6, 0.30])

        axNextImg = fig.add_axes([0.72, 0.65, 0.25, 0.10])
        axPrevImg = fig.add_axes([0.72, 0.53, 0.25, 0.10])
        axNextStat = fig.add_axes([0.72, 0.25, 0.25, 0.10])
        axPrevStat = fig.add_axes([0.72, 0.13, 0.25, 0.10])

        fig.text(
            0.98, 0.98,
            f"Total processing time: {totalTime:.2f} s",
            ha="right", va="top",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.7),
            color="white"
        )

        # -------- IMAGE --------
        def drawImage():
            axImage.clear()
            data = results[currentImageIndex[0]]

            axImage.imshow(data["image"], cmap="gray")
            axImage.axis("off")


            for (x, y, w, h), label, gtLabel in zip(
                data["boxes"],
                data["labels"],
                data["gtLabels"]
            ):
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
                    f"{label}",
                    color="white",
                    fontsize=6,
                    bbox=dict(facecolor=color, alpha=0.85)
                )


            axImage.set_title(f"Image {currentImageIndex[0] + 1}")
            fig.canvas.draw_idle()

        # -------- METRICS --------
        def drawMetrics():
            axStats.clear()

            values = [accuracy, precision, recall, f1Score]
            labels = [
                f"Accuracy {accuracy:.2f}",
                f"Precision {precision:.2f}",
                f"Recall {recall:.2f}",
                f"F1 {f1Score:.2f}"
            ]

            axStats.bar(labels, values)
            axStats.set_ylim(0, 1)
            axStats.set_title("Final metrics (10k test)")

            axImage.set_title(f"Image {currentImageIndex[0] + 1}")

            fig.canvas.draw_idle()

        def drawConfusionMatrix():
            axStats.clear()
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axStats,
                cbar=False
            )
            axStats.set_title("Confusion Matrix (10k test)")
            axStats.set_xlabel("Predicted")
            axStats.set_ylabel("True")

        statsFns = [drawMetrics, drawConfusionMatrix]


        def drawStats():
            statsFns[currentStatsIndex[0]]()
            fig.canvas.draw_idle()

        # -------- BUTTONS --------
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


        btnNextImg = Button(axNextImg, "Next image")
        btnPrevImg = Button(axPrevImg, "Previous image")

        btnNextStat = Button(axNextStat, "Next stats")
        btnPrevStat = Button(axPrevStat, "Previous stats")

        btnNextImg.on_clicked(onNextImage)
        btnPrevImg.on_clicked(onPrevImage)

        btnNextStat.on_clicked(onNextStats)
        btnPrevStat.on_clicked(onPrevStats)

        drawImage()
        drawStats()
        plt.show()


    
    def plotLossCurves(self):
        epochs = range(1, len(self.trainLossHistory) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.trainLossHistory, "r-", label="Train")
        plt.plot(epochs, self.testLossHistory, "b-", label="Test")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("loss_vs_epochs.png")
        plt.close()


    def train(self):
        for epoch in range(self.args["numEpochs"]):
            trainLoss = self.trainEpoch(epoch)
            testLoss = self.evaluate(epoch)

            self.trainLossHistory.append(trainLoss)
            self.testLossHistory.append(testLoss)

            print(
                f"Epoch [{epoch + 1}/{self.args['numEpochs']}] "
                f"Train Loss: {trainLoss:.4f} "
                f"Test Loss: {testLoss:.4f}"
            )

        print("Training finished.")
        self.plotLossCurves()
        self.evaluateFinal()

        torch.save(self.model.state_dict(), "final_model_10k.pth")
        print("Final model saved as final_model_10k.pth")

