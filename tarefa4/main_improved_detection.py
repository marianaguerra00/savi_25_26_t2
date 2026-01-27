import argparse
import torch
from dataset import getImprovedDatasets
from model import ModelImprovedDetector, ModelImprovedDetectorV2, ModelImprovedDetectorLite
from trainer import ImprovedTrainer

def main():
    parser = argparse.ArgumentParser(
        description="Tarefa 4 - Improved Digit Detector & Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        required=True,
        help="train: train and test | test: test only with saved model"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["lite", "v1", "v2"],
        default="lite",
        help="lite: Lightweight ~300K params | v1: FCN ~1.5M params | v2: ResNet ~2M params"
    )
    
    parser.add_argument(
        "--modelPath",
        type=str,
        default="final_model.pth",
        help="Path to saved model (for test mode)"
    )
    
    parser.add_argument(
        "--dataPath",
        type=str,
        default="../improved_dataset/versionD",
        help="Path to dataset directory"
    )
    
    parser.add_argument(
        "--batchSize",
        type=int,
        default=64,
        help="Batch size for training/testing"
    )
    
    parser.add_argument(
        "--numEpochs",
        type=int,
        default=15,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learningRate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("TAREFA 4: Integrated Detector and Classifier")
    print("="*60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Model: {args.model.upper()}")
    print(f"Batch size: {args.batchSize}")
    if args.mode == "train":
        print(f"Epochs: {args.numEpochs}")
        print(f"Learning rate: {args.learningRate}")
    print("="*60 + "\n")
    
    # ----------------------------
    # Load datasets
    # ----------------------------
    print("Loading datasets...")
    trainDataset, testDataset = getImprovedDatasets(args.dataPath)
    print(f"✓ Train samples: {len(trainDataset)}")
    print(f"✓ Test samples: {len(testDataset)}\n")
    
    # ----------------------------
    # Create model
    # ----------------------------
    print(f"Creating model ({args.model})...")
    if args.model == "lite":
        model = ModelImprovedDetectorLite(numClasses=10)
        print("✓ Using Lightweight architecture (~300K params)")
    elif args.model == "v1":
        model = ModelImprovedDetector(numClasses=10)
        print("✓ Using FCN with BatchNorm architecture (~1.5M params)")
    else:
        model = ModelImprovedDetectorV2(numClasses=10)
        print("✓ Using ResNet-inspired architecture (~2M params)")
    
    print(f"✓ Parameters: {model.count_parameters():,}\n")
    
    # ----------------------------
    # Create trainer
    # ----------------------------
    trainer = ImprovedTrainer(
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
        trainer.train()
    
    # ----------------------------
    # TEST MODE
    # ----------------------------
    elif args.mode == "test":
        print(f"Loading model from: {args.modelPath}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            model.load_state_dict(torch.load(args.modelPath, map_location=device))
            model.to(device)
            print("✓ Model loaded successfully\n")
            trainer.evaluateFinal()
        except FileNotFoundError:
            print(f"Error: Model file '{args.modelPath}' not found!")
            print("Please train a model first or specify correct path.")
            return


if __name__ == "__main__":
    main()