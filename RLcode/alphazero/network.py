import torch
from torch import nn
import torch.nn.functional as F

from .functional import sparse_cross_entropy

DEVICE = 'cpu'

class TorchBaseModel(nn.Module):
    """
    Base model for value and policy estimation.
    """
    def forward(self, x):
        x = self.network(x)
        value = torch.tanh(x[:, 0].unsqueeze(1))
        policy_scores = x[:, 1:]
        return value, policy_scores

class ResidualBlock(nn.Module):
    """
    Residual block of 3 linear layers.
    """
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.block = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.BatchNorm1d(self.n_features),
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.BatchNorm1d(self.n_features),
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.BatchNorm1d(self.n_features),
        )
    
    def forward(self, x):
        r = self.block(x)
        x = x + r
        return x

class TTT_model(TorchBaseModel):
    """
    Model used for game Tic Tac Toe.
    """
    def __init__(self, input_size=18, action_space_size=9):
        super().__init__()
        self.input_size = input_size
        self.action_space_size = action_space_size
        self.network = nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Linear(64, 1 + self.action_space_size),
        )

class C4_model(TorchBaseModel):
    """
    Model used for game Connect 4.
    """
    def __init__(self, action_space_size=7):
        super().__init__()
        self.action_space_size = action_space_size
        self.network = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128, 1 + self.action_space_size)
        )