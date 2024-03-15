import torch
import torch.nn as nn
from .networks import Res18_VAE

class base_Model(nn.Module):
    """Builds the neural network."""
    def __init__(self, configs, device):
        super().__init__()
        
        self.net = Res18_VAE(configs)
        
        self.center = None
        self.length = torch.tensor(0, device=device)

    def forward(self, x):
        
        return self.net(x)