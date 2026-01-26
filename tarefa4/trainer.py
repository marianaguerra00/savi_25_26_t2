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
    
    def evaluateFinal(self):
        self.model.eval()

        allTrueLabels = []
        allPredLabels = []

        with torch.no_grad():
            for images, confT, classT, bboxT in self.testLoader:
                images = images.to(self.device)
                confT = confT.to(self.device)
                classT = classT.to(self.device)

                outputs = self.model(images)

                classPred = outputs[:, 1:11]  # [B, 10, H, W]
                confPred = torch.sigmoid(outputs[:, 0])  # [B, H, W]

                # Only consider cells where an object exists
                objectMask = confT > 0.5

                if objectMask.sum() == 0:
                    continue

                classPredFlat = classPred.permute(0, 2, 3, 1)[objectMask]
                classTrueFlat = classT[objectMask]

                predictedLabels = torch.argmax(classPredFlat, dim=1)

                allTrueLabels.extend(classTrueFlat.cpu().numpy())
                allPredLabels.extend(predictedLabels.cpu().numpy())

        # Metrics
        accuracy = accuracy_score(allTrueLabels, allPredLabels)
        precision = precision_score(allTrueLabels, allPredLabels, average="macro")
        recall = recall_score(allTrueLabels, allPredLabels, average="macro")
        f1 = f1_score(allTrueLabels, allPredLabels, average="macro")

        print("\nFINAL TEST RESULTS (10k images)")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(allTrueLabels, allPredLabels))

        # Confusion Matrix
        cm = confusion_matrix(allTrueLabels, allPredLabels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix (Final Test)")
        plt.tight_layout()
        plt.savefig("confusion_matrix_final.png")
        plt.close()

    
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

