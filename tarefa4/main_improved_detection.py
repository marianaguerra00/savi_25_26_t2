import argparse
import torch

from dataset import getImprovedDatasets
from model import ModelImprovedDetector
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Tarefa 4 - Detector de Digitos")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        required=True,
        help="train: treina e testa | test: apenas testa modelo guardado"
    )

    parser.add_argument(
        "--modelPath",
        type=str,
        default="final_model_10k.pth",
        help="Caminho para o modelo .pth (usado em mode=test)"
    )

    parser.add_argument(
        "--batchSize",
        type=int,
        default=64
    )

    parser.add_argument(
        "--numEpochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--learningRate",
        type=float,
        default=1e-3
    )

    args = parser.parse_args()

    # ----------------------------
    # Load datasets
    # ----------------------------
    trainDataset, testDataset = getImprovedDatasets()

    # ----------------------------
    # Create model
    # ----------------------------
    model = ModelImprovedDetector(numClasses=10)

    # ----------------------------
    # Trainer
    # ----------------------------
    trainer = Trainer(
        model=model,
        trainDataset=trainDataset,
        testDataset=testDataset,
        args={
            "batchSize": args.batchSize,
            "numEpochs": args.numEpochs,
            "learningRate": args.learningRate
        }
    )

    # ----------------------------
    # TRAIN MODE
    # ----------------------------
    if args.mode == "train":
        print("Mode: TRAIN + FINAL TEST")
        trainer.train()

    # ----------------------------
    # TEST MODE
    # ----------------------------
    elif args.mode == "test":
        print(f"Mode: TEST ONLY | Loading model from {args.modelPath}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(args.modelPath, map_location=device))
        model.to(device)

        trainer.evaluateFinal()


if __name__ == "__main__":
    main()
