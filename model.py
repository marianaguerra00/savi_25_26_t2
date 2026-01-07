import torch # importando o pytorch
import torch.nn as nn # importando o modulo de redes neurais do pytorch
import torch.nn.functional as F # importando funcoes de ativacao e outras funcoes uteis

class Model(nn.Module): 

    def __init__ (self):
        super(Model, self).__init__() # chamada do construtor da classe mae nn.Module

        nrows = 28
        ncols = 28 # dimensao das imagens do dataset MNIST
        ninputs = nrows * ncols # numero de entradas da rede neural
        noutputs = 10 

        # definicao das camadas da rede neural
        self.fc = nn.Linear(ninputs, noutputs) 

        print('model architecture initialized with ' + str(self.getNumberOfParameters()) + ' parameters.')

    def forward(self, x):

        # x_og = 4 dimensões: Batch, Canal, Altura, Largura || flatten the input to a vector of 1x28x28
        x = x.view(x.size(0), -1)

        # now we can pass through the fully connected layer
        y = self.fc(x)

        return y
    
        # Cada um dos 10 valores em y representa a pontuação bruta que a rede atribui à probabilidade de a imagem de entrada pertencer à classe 
        # correspondente (dígito 0 a 9). Estes valores podem ser positivos ou negativos e não somam 1.

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        

class ModelBetterCNN(nn.Module):
    def __init__(self):
        super(ModelBetterCNN, self).__init__()
        
        # Bloco 1: Detalhes finos
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Bloco 2: Formas intermédias
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Bloco 3: Formas complexas
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Classificador (FC)
        # Dimensões: 28 -> 14 -> 7 -> 3 (Devido a 3 Poolings)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # x: [Batch, 1, 28, 28]
        
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # -> [32, 14, 14]
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # -> [64, 7, 7]
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # -> [128, 3, 3]
        
        x = x.view(-1, 128 * 3 * 3) # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Retorna Logits
        return x
    
    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)