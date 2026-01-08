import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import seaborn
import json 
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

class Trainer():
    def __init__(self, model, dataset_train, dataset_test, args):
        # 1. Store parameters
        self.model = model
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.args = args 

        # 2. Define Loss Function & Optimizer
        # CrossEntropyLoss combina LogSoftmax + NLLLoss (requer Logits como input)
        self.loss = nn.CrossEntropyLoss() 
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # 3. Initialize state
        self.history = {'train_loss': []}
        self.train_epoch_losses = []
        self.test_epoch_losses = []
        self.epoch_idx = 0

        # 4. DataLoaders
        self.dataloader_train = DataLoader(
            dataset=dataset_train,
            batch_size=args['batch_size'],
            shuffle=True,
            num_workers=4
        )

        self.dataloader_test = DataLoader(
            dataset=dataset_test,
            batch_size=args['batch_size'],
            shuffle=False,
            num_workers=4
        )

        # 5. GPU Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.model.to(self.device)

        # 6. Resume Training Logic
        if self.args['resume_training']:
            self.loadTrain()

        # Setup plot
        plt.title("Training Loss vs epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        self.axis = plt.gca()
        self.axis.set_xlim([1, self.args['num_epochs']+1]) 
        self.axis.set_ylim([0, 0.5]) # Ajustado para escala da CrossEntropy

    def train(self): 
        num_epochs = self.args['num_epochs']

        for i in range(self.epoch_idx, num_epochs):
            
            # --- TRAIN LOOP ---
            self.model.train() 
            train_batch_losses = []

            print(f"\n--- Época {i+1}/{num_epochs} ---")
    
            for batch_idx, (images, labels) in tqdm(enumerate(self.dataloader_train), total=len(self.dataloader_train), desc=f'Treino'): 
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward (Logits)
                logits = self.model.forward(images)

                # Loss
                batch_loss = self.loss(logits, labels)
                train_batch_losses.append(batch_loss.item())

                # Backprop
                self.optimizer.zero_grad() 
                batch_loss.backward() 
                self.optimizer.step()
                    
            
            # --- VALIDATION LOOP ---
            self.model.eval() 
            test_batch_losses = []

            with torch.no_grad(): 
                for batch_idx, (images, labels) in tqdm(enumerate(self.dataloader_test), total=len(self.dataloader_test), desc=f'Teste'):
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    logits = self.model.forward(images)
                    batch_loss = self.loss(logits, labels)
                    test_batch_losses.append(batch_loss.item())

            # Update Metrics
            train_epoch_loss = np.mean(train_batch_losses)
            self.train_epoch_losses.append(train_epoch_loss)

            test_epoch_loss = np.mean(test_batch_losses)
            self.test_epoch_losses.append(test_epoch_loss)
            
            print(f"Época {i+1}: Loss Treino = {train_epoch_loss:.4f} | Loss Teste = {test_epoch_loss:.4f}")
            self.draw() 

        print(f"\n--- Treino Concluído ({num_epochs} Épocas) ---")

    def evaluate(self):
        print("\n--- A Iniciar Avaliação Final ---")

        self.model.eval() 
        gt_classes = []
        predicted_classes = []

        with torch.no_grad():
            for batch_idx, (images, labels) in tqdm(enumerate(self.dataloader_test), total=len(self.dataloader_test), desc='A processar'):

                images = images.to(self.device)
                labels = labels.to(self.device)

                # Labels reais
                batch_gt_classes = labels.tolist()

                # Previsões
                logits = self.model.forward(images)
                probabilities = torch.softmax(logits, dim=1)
                batch_predicted_classes = probabilities.argmax(dim=1).tolist()

                gt_classes.extend(batch_gt_classes)
                predicted_classes.extend(batch_predicted_classes)

        # ----------------------
        # 1. Matriz de Confusão
        # ----------------------
        cm = confusion_matrix(gt_classes, predicted_classes)
        
        plt.figure(figsize=(10, 8))
        class_names = [str(i) for i in range(10)]
        seaborn.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

        # ----------------------
        # 2. Relatório Scikit-Learn
        # ----------------------
        report = classification_report(gt_classes, predicted_classes, digits=4)
        print("\nRelatório de Classificação:")
        print(report)

        acc = accuracy_score(gt_classes, predicted_classes)
        print(f"\nAccuracy Global: {acc*100:.2f}%")

        # Guardar JSON
        report_dict = classification_report(gt_classes, predicted_classes, output_dict=True)
        json_filename = os.path.join(self.args['experiment_full_name'], 'statistics.json')
        with open(json_filename, 'w') as f:
            json.dump(report_dict, f, indent=4)
        print(f"Resultados guardados em {json_filename}")

    def draw(self):
        xs = range(1, len(self.train_epoch_losses)+1)
        plt.plot(xs, self.train_epoch_losses, 'r-', linewidth=2, label='Train')
        plt.plot(xs, self.test_epoch_losses, 'b-', linewidth=2, label='Test')
        plt.legend()
        plt.savefig('training.png')
        plt.close() # Limpar figura

    def saveTrain(self):
        checkpoint = {
            'epoch_idx': self.epoch_idx, # Corrigido typo 'epock_idk'
            'train_epoch_losses': self.train_epoch_losses,
            'test_epoch_losses': self.test_epoch_losses,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        checkpoint_file = os.path.join(self.args['experiment_full_name'], 'checkpoint.pkl')
        torch.save(checkpoint, checkpoint_file)

    def loadTrain(self):
        print('Resuming training from last checkpoint...')
        checkpoint_file = os.path.join(self.args['experiment_full_name'], 'checkpoint.pkl')
        
        if not os.path.exists(checkpoint_file):
            raise ValueError('Checkpoint file not found: ' + checkpoint_file)
        
        checkpoint = torch.load(checkpoint_file)
        
        self.epoch_idx = checkpoint['epoch_idx']
        self.train_epoch_losses = checkpoint['train_epoch_losses']
        self.test_epoch_losses = checkpoint['test_epoch_losses']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])