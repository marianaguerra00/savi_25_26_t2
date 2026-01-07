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

        #1. store parameters
        self.model = model
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.args = args #dictionary with terminal arguments

        #2. define loss function
        # we are using BCEWithLogitsLoss because the output doesnr't have a softmax layer and label is OHE
        #self.loss = nn.MSELoss()

        self.loss = nn.CrossEntropyLoss() 
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        #3. intialize state variables (for loss)
        self.history = {'train_loss': []}

        # ------------------------------------
        # Ex3.b implementar dataloaders
        # ------------------------------------

        self.dataloader_train = DataLoader(
            dataset=dataset_train,
            batch_size=args['batch_size'],
            shuffle=True,                   # baralha os dados de treino a cada época
            num_workers=4                   # 0 p debbug, >0 p performance
        )

        self.dataloader_test = DataLoader(
            dataset=dataset_test,
            batch_size=args['batch_size'],
            shuffle=False,                  # os dados teste necessitam de estar ordenados
            num_workers=4                   
        )

        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=0.0001 # Learning Rate que definimos
        )

        # ------------------------------------
        # configurar treino na GPU
        # ------------------------------------

        # 1. Definir o dispositivo: Usar CUDA (GPU) se disponível, senão CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        # 2. Mover o modelo para a GPU
        self.model.to(self.device)

        # ------------------------------------

        # define loss for epoch
        self.train_epoch_losses = []
        self.test_epoch_losses = []

        if self.args['resume_training']:
            self.loadTrain()
        else:
            self.train_epoch_losses = []
            self.test_epoch_losses = []
            self.epoch_idx = 0

        # setup figure for plotting
        plt.title("Training Loss vs epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        axis = plt.gca()
        axis.set_xlim([1, self.args['num_epochs']+1])  # type: ignore
        axis.set_ylim([0, 0.2])  # type: ignore

    def train(self): 
        num_epochs = self.args['num_epochs']

        for i in range(self.epoch_idx, num_epochs):
            
            self.model.train() 
            train_batch_losses = []

            print(f"\n--- Época {i+1}/{num_epochs} ---")
    
            # --- LOOP DE TREINO ---
            for batch_idx, (images, labels) in tqdm(enumerate(self.dataloader_train), total=len(self.dataloader_train), desc=f'Treino época {i+1}'): 
                
                images = images.to(self.device)
                labels = labels.to(self.device) # Labels são índices (ex: 5)
                
                # 1. Forward (Recebe Logits)
                logits = self.model.forward(images)

                # 2. Loss (CrossEntropy: Logits vs Índices)
                # NÃO APLICAR SOFTMAX AQUI!
                batch_loss = self.loss(logits, labels)
                
                train_batch_losses.append(batch_loss.item())

                # 3. Update model
                self.optimizer.zero_grad() 
                batch_loss.backward() 
                self.optimizer.step()
                    
            
            # --- LOOP DE VALIDAÇÃO ---
            self.model.eval() 
            test_batch_losses = []

            with torch.no_grad(): 
                for batch_idx, (images, labels) in tqdm(enumerate(self.dataloader_test), total=len(self.dataloader_test), desc=f'Teste época {i+1}'):
                    
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Forward
                    logits = self.model.forward(images)

                    # Loss
                    batch_loss = self.loss(logits, labels)
                    test_batch_losses.append(batch_loss.item())

            # update losses
            train_epoch_loss = np.mean(train_batch_losses)
            self.train_epoch_losses.append(train_epoch_loss)

            test_epoch_loss = np.mean(test_batch_losses)
            self.test_epoch_losses.append(test_epoch_loss)
            
            print(f"Época {i+1}: Loss Treino = {train_epoch_loss:.4f} | Loss Teste = {test_epoch_loss:.4f}")
            self.draw() 

        print(f"\n--- Treino Concluído ({num_epochs} Épocas) ---")

    def saveTrain(self):

        #create the dictionary to save the checkpoint.pkl
        checkpoint = {}
        checkpoint['epock_idk']=self.epoch_idx
        checkpoint['train_epoch_losses']=self.train_epoch_losses
        checkpoint['test_epoch_losses']=self.test_epoch_losses

        checkpoint['model_state_dict']=self.model.state_dict()
        checkpoint['optimizer_state_dict']=self.optimizer.state_dict()

        checkpoint_file = os.path.join(self.args['experiment_full_name'], 'checkpoint.pkl')
        torch.save(checkpoint, checkpoint_file)

    def loadTrain(self):
        
        print('Resuming training from last checkpoint...')

        checkpoint_file = os.path.join(self.args['experiment_full_name'], 'checkpoint.pkl')
        print('Loading checkpoint from: ' + checkpoint_file)

        if not os.path.exists(checkpoint_file):
            raise ValueError('Checkpoint file not found: ' + checkpoint_file)
        
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        print(checkpoint.keys())

        self.epoch_idx = checkpoint['epock_idx']
        self.train_epoch_losses = checkpoint['train_epoch_losses']
        self.test_epoch_losses = checkpoint['test_epoch_losses']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def draw(self):

        # plot training
        xs = range(1, len(self.train_epoch_losses)+1)
        ys = self.train_epoch_losses
        plt.plot(xs, ys, 'r-', linewidth=2)

        # plot testing
        xs = range(1, len(self.test_epoch_losses)+1)
        ys = self.test_epoch_losses
        plt.plot(xs, ys, 'b-', linewidth=2)

        plt.legend(['Train', 'Test'])

        plt.savefig('training.png')

    def evaluate(self):
        print("\n--- A Iniciar Avaliação ---")

        self.model.eval() 
        gt_classes = []
        predicted_classes = []

        with torch.no_grad():
            for batch_idx, (images, labels) in tqdm(enumerate(self.dataloader_test), total=len(self.dataloader_test), desc='A processar'):

                images = images.to(self.device)
                labels = labels.to(self.device)

                batch_gt_classes = labels.tolist()

                logits = self.model.forward(images)
                probabilities = torch.softmax(logits, dim=1)
                batch_predicted_classes = probabilities.argmax(dim=1).tolist()

                gt_classes.extend(batch_gt_classes)
                predicted_classes.extend(batch_predicted_classes)

        # ----------------------
        # 1. Matriz de Confusão (Visual)
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
        # 2. Relatório Completo (Substitui o cálculo manual)
        # ----------------------
        # Isto calcula Precision, Recall, F1-Score e Support para todas as classes
        # e ainda dá as médias Macro e Weighted.
        
        report = classification_report(gt_classes, predicted_classes, digits=4)
        
        print("\nRelatório de Classificação (sklearn):")
        print(report)

        # Cálculo da Accuracy exata
        acc = accuracy_score(gt_classes, predicted_classes)
        print(f"\nAccuracy Global: {acc*100:.2f}%")

        # Guardar estatísticas em JSON (Opcional: O report do sklearn pode sair como dict)
        report_dict = classification_report(gt_classes, predicted_classes, output_dict=True)
        json_filename = os.path.join(self.args['experiment_full_name'], 'statistics.json')
        with open(json_filename, 'w') as f:
            json.dump(report_dict, f, indent=4)
        print(f"Estatísticas guardadas em {json_filename}")

    def getPrecisionRecall(self, TPs, FPs, FNs):

        den = TPs + FPs
        if den == 0:
            precision = None
        else:
            precision = TPs/(TPs+FPs)

        den = TPs + FNs
        if den == 0:
            recall = None
        else:
            recall = TPs/(TPs+FNs)

        return precision, recall
        
        