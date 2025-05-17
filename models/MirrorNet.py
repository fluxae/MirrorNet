import torch
import torch.nn as nn
from layers.RevIN import RevIN
import torch.nn.functional as F
import math
import numpy as np

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.scale = 0.02
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)  

        self.embed_size = self.seq_len

        self.householder_num = configs.householder_num

        self.householder_vecs = nn.Parameter(self.scale * torch.randn(self.householder_num, 2 * (self.embed_size//2 + 1)))
        
        self.basis_len = self.embed_size

        self.fc =  nn.Sequential(
                                            nn.Linear(self.basis_len * (self.embed_size // 2 + 1), 3*720),      
                                            nn.Dropout(p=0.15),
                                            nn.ReLU(),
                                            nn.Linear(3*720, 3*720),
                                            nn.Dropout(p=0.15),
                                            nn.ReLU(),
                                            nn.Linear(3*720, self.pred_len)
                                        )
        
        self.fc_simple = nn.Sequential(
                                            nn.Linear(2 * (self.embed_size // 2 + 1), 256),      
                                            nn.ReLU(),
                                            nn.Linear(256, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, self.pred_len)
                                        )
    
    def plain_upsampling(self, x):
        return x.repeat(1, 1, self.basis_len)
 
    def householder_layers(self, x, householder_vecs):
        householder_vecs = householder_vecs / (torch.norm(householder_vecs, p = 2, dim = -1).unsqueeze(-1))       
        for i in range(householder_vecs.size()[0]):
            householder_vec = householder_vecs[i]
            x = x - 2 * torch.sum(x * householder_vec, dim=-1, keepdim=True) * householder_vec  
        return x
    
    def householder_mlp_inner(self, x):
        x = torch.fft.rfft(x, dim = -1, norm='ortho')    
        x = torch.cat([x.real, x.imag], dim = -1)
        x = self.householder_layers(x, self.householder_vecs)
        basis_cos = self.plain_upsampling(x[: ,:, :(self.embed_size // 2 + 1)])
        basis_sin = self.plain_upsampling(x[: ,:, (self.embed_size // 2 + 1):])
        x = basis_cos + basis_sin
        x = self.fc(x)
        return x
    
    def householder_mlp_inner_simple(self, x):
        x = torch.fft.rfft(x, dim = -1, norm='ortho')    
        x = torch.cat([x.real, x.imag], dim = -1)
        x = self.householder_layers(x, self.householder_vecs)
        x = self.fc_simple(x)
        return x
        
    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        z = x
        z = self.revin_layer(z, 'norm')
        x = z

        x = x.permute(0, 2, 1)
        x = self.householder_mlp_inner(x)
        #x = self.householder_mlp_inner_simple(x)
        x = x.permute(0, 2, 1)

        z = x
        z = self.revin_layer(z, 'denorm')
        x = z

        

        return x
    