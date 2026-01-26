from dataset import getImprovedDatasets
from model import ModelImprovedDetector
from trainer import Trainer


args = {
    "batchSize": 64,
    "numEpochs": 10,
    "learningRate": 1e-3
}

trainDataset, testDataset = getImprovedDatasets()

model = ModelImprovedDetector(numClasses=10)

trainer = Trainer(
    model=model,
    trainDataset=trainDataset,
    testDataset=testDataset,
    args=args
)

trainer.train()
