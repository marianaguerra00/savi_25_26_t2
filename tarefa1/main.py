"""
Trabalho Prático 2 - Tarefa 1: Classificador CNN Otimizado (MNIST Completo)
"""

import argparse
import os
import shutil
import signal
import sys
from datetime import datetime

# Imports dos módulos desenvolvidos
from dataset import get_mnist_datasets
from model import ModelBetterCNN
from trainer import Trainer
import random
import numpy as np
import torch

def set_seed(seed=1234):
    """Torna o treino determinístico para garantir reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Garante que o cuDNN (aceleração GPU) é determinístico (pode ser mais lento)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    print(f"Random Seed fixada em: {seed}")

# No main block:
if __name__ == "__main__":
    set_seed(42) # Podes testar outros números (0, 1, 123) até voltares aos 99.4%
    # ... resto do código ...
def sigintHandler(signum, frame):
    print('SIGINT received. Exiting gracefully.')
    sys.exit(0)

def main():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "datasets", "mnist_aapi")
    EXPERIMENT_PATH = os.path.join(BASE_DIR, "experiments", "savi_experiments")

    parser = argparse.ArgumentParser()
    parser.add_argument('-df', '--dataset_folder', type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-ne', '--num_epochs', type=int, default=10)
    parser.add_argument('-pe', '--percentage_examples', type=float, default=0.2,
                        help="Percentage of examples to use for training and testing (0-1)")
    parser.add_argument('-ep', '--experiment_path', type=str, default=EXPERIMENT_PATH,
                        help="Path to save experiment results")
    parser.add_argument('-rt', '--resume_training', action='store_true',
                        help='Resume training from last checkpoint if available.')

    args = vars(parser.parse_args())
    print(args)
    
    # ------------------------------------
    # register the sigtint handler
    # ------------------------------------
    signal.signal(signal.SIGINT, sigintHandler)

    # ------------------------------------
    # 0. Create Experiments Folder
    # ------------------------------------

    experiment_name = datetime.now().strftime("%Y-%m-%d %H")

    args['experiment_full_name'] = os.path.join(args['experiment_path'], experiment_name)

    print('starting experiment: ' + args['experiment_full_name'])

    if os.path.exists(args['experiment_full_name']):
        shutil.rmtree(args['experiment_full_name'])
        print('Experiment folder already exists. Deleting it to start fresh.')

    os.makedirs(args['experiment_full_name'])

    # ------------------------------------
    # 1. Create Datasets (Train and Test!)
    # ------------------------------------
    print("A carregar dataset...")
    dataset_train, dataset_test = get_mnist_datasets()

    # ------------------------------------
    # 2. Create Model
    # ------------------------------------
    model = ModelBetterCNN()
    print(f"Modelo carregado com {model.getNumberOfParameters()} parâmetros.")

    # ------------------------------------
    # 3. Create Trainer
    # ------------------------------------
    trainer = Trainer(model, dataset_train, dataset_test, args)

    # ------------------------------------
    # 4. Execute Training and Evaluation
    # ------------------------------------
    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()