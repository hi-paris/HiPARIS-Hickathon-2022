"""
    Script of the carClassifier
"""

from torchvision import models, transforms
from torch import nn, optim
import torch

class carClassifier(torch.nn.Module):
    def __init__(self, n_labels=100, scaler=True):
        super().__init__()

        pretrained = models.resnet50(pretrained=True)
        backbone = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
        for x in backbone.parameters():
            x.requires_grad = False
            
        for x in list(backbone.parameters())[-3:-1]:
            x.requires_grad = True
        
        self.scaler = transforms.Lambda(lambda x: x/255.)
        self.network = nn.Sequential(*[
            backbone,
            nn.Flatten(),
            nn.Linear(2048, n_labels)
        ])

        self.optimizer = optim.Adam(self.parameters(), lr=5e-3)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.scaler(x)
        x = self.network(x)

        return x

    def predict(self, x):
        self.eval()

        with torch.no_grad():
            y_hat = self.forward(x)

        return y_hat

    def fit(self, x, y):

        self.train()
        self.optimizer.zero_grad()

        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        loss.backward()
        self.optimizer.step()

        return loss.detach().item()