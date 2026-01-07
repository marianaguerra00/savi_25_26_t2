"""
objetivo: construir, treinar e avaliar uma rede neural simples para 
classifição de dígitos manuscritos utilizando o dataset MNIST com PyTorch.

Parte 1: classe Dataset, responsável por carregar e preparar os dados
"""

### Construção de um leitor de datasets com a classe Dataset

import argparse
import glob
import os
from random import randint
from matplotlib import pyplot as plt
import numpy as np
from dataset import CustomDataset
from dataset import get_mnist_datasets
from model import Model
from model import ModelBetterCNN
from trainer import Trainer
import wandb 
from datetime import datetime
import shutil
import signal 
import sys

def sigintHandler(signum, frame):
    print('SIGINT received. Exiting gracefully.')
    sys.exit(0)

def main():

    # -----------------
    # Setup argparse
    # -----------------

    # caminho correto para datasets
    DEFAULT_DATASET_PATH = r"C:\Users\maria\Desktop\datasets\mnist_aapi"
    EXPERIMENT_PATH = r"C:\Users\maria\Desktop\datasets\savi_experiments"

    parser = argparse.ArgumentParser()
    parser.add_argument('-df', '--dataset_folder', type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-ne', '--num_epochs', type=int, default=10)
    parser.add_argument('-pe', '--percentage_examples', type=float, default=0.2, help="Percentage of examples to use for training and testing (0-1)")
    parser.add_argument('-ep', '--experiment_path', type=str, default=EXPERIMENT_PATH, help="Path to save experiment results")
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
    # dataset_train = Dataset(args, is_train=True)
    # dataset_test = Dataset(args, is_train=False)

    dataset_train, dataset_test = get_mnist_datasets()

    # ------------------------------------
    # 2. Create Model
    # ------------------------------------
    #model = Model()
    model = ModelBetterCNN()
    print(f"Modelo carregado com {model.getNumberOfParameters()} parâmetros.")

    # ------------------------------------
    # 3. Create Trainer
    # ------------------------------------
    trainer = Trainer(model, dataset_train, dataset_test, args)

    # ------------------------------------
    # 4. Start Training
    # ------------------------------------
    trainer.train()

    trainer.evaluate()
    

    # wandb.login()

    # # Project that the run is recorded to
    # project = "my-awesome-project"

    # # Dictionary with hyperparameters
    # config = {
    #     'epochs' : 10,
    #     'lr' : 0.01
    # }

    # with wandb.init(project=project, config=config) as run:
    #     trainer.train()

    #     trainer.evaluate()
        
    #     run.log({"accuracy": 0.9, "loss": 0.1})

if __name__ == "__main__":
    main()