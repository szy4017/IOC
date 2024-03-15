import torch
import torch.nn as nn
from .inn import INN

class base_Model(nn.Module):
    """Builds the neural network."""
    def __init__(self, configs, device):
        super().__init__()
        
        self.d_model = configs.d_model
        self.d_res = configs.d_res
        self.res_con = configs.res_con

        self.enc = nn.Conv2d(3, configs.d_model, kernel_size=3, stride=1, padding=1, bias=False)
        self.inv_net = INN(configs.d_model, configs.d_res, configs.link, configs.num_inv, configs.clamp, device)
        # self.dec_net = Decoder(configs)
        self.dec = nn.Conv2d(configs.d_model, 3, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        emb = self.enc(x) # (B, D, W, H)
        emb_a = emb[:, :-self.d_res]
        emb_b = emb[:, -self.d_res:]

        emb_rec, emb_con = self.inv_net(emb_a, emb_b, rev=False)
        rec_a, rec_b = self.inv_net(emb_rec, torch.ones_like(emb_con)*self.res_con, rev=True)
        # emb_rec: (B, D1, H, W), emb_con: (B, D2, H, W)
        rec_ab = torch.cat((rec_a, rec_b),dim=1) # (B, D, H, W)
        # for_rec = self.dec_net(emb_rec)
        inv_rec = self.dec(rec_ab)
        
        return inv_rec, emb_con