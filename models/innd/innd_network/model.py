import torch
import torch.nn as nn
from .networks import HSR_LeNet, HSR_ResNet18, Res18_VAE

class base_Model(nn.Module):
    """Builds the neural network."""
    def __init__(self, configs, device):
        super().__init__()
        
        if configs.net_name == 'hsr_LeNet':
            self.net = HSR_LeNet(configs)
        if configs.net_name == 'hsr_res18':
            self.net = HSR_ResNet18(configs)
        if configs.net_name == 'res18_vae':
            self.net = Res18_VAE(configs)
        
        self.center = None
        self.length = torch.tensor(0, device=device)

    def forward(self, x):
        
        
        return self.net(x)