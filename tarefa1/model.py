import torch # importando o pytorch
import torch.nn as nn # importando o modulo de redes neurais do pytorch
import torch.nn.functional as F # importando funcoes de ativacao e outras funcoes uteis

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
        self.dropout = nn.Dropout(0.8)
        
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