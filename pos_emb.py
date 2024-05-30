import torch
import torch.nn as nn

import math

class PositionalEmbedding(nn.Module):

   def __init__(self, length=6, emb=300, device='cpu',*args, **kwargs):
      super().__init__(*args, **kwargs)
      self.emb = emb
      self.length = length
      self.device = device
      self.register_buffer('pos_emb', self._make_pos_emb())

   def _make_pos_emb(self):
      pos_emb = torch.zeros(self.length, self.emb).to(self.device)
      
      position = torch.arange(0, self.length).unsqueeze(1) 
      div_term = torch.exp(torch.arange(0, self.emb, 2) * (-math.log(10_000) / self.emb))

      pos_emb[:, 0::2] = torch.sin(position * div_term)
      pos_emb[:, 1::2] = torch.cos(position * div_term)
      return nn.Parameter(pos_emb, requires_grad=False)

   @torch.no_grad()
   def forward(self, x):
      return x + self.pos_emb
   

if __name__ == '__main__':
   pos = PositionalEmbedding()
   
   x = torch.ones(50, 6, 300)
   assert pos(x).shape == (50, 6, 300)
   