# generator_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 10, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_onehot = F.one_hot(labels, num_classes=10).float()
        x = torch.cat((z, label_onehot), dim=1)
        return self.model(x).view(-1, 1, 28, 28)