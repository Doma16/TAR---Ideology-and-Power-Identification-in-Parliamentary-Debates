import torch
import torch.nn as nn

import math

class RTransformer(nn.Module):
   
   def __init__(self, *args,
               emb_dim=300,
               nhead=4,
               num_layers=4,
               k=6,
               stride=4,
               **kwargs):
      super(RTransformer, self).__init__(*args, **kwargs)
      self.encoder = TransformerEncoder(emb_dim, nhead, num_layers, k, dropout=0.1)
      self.reset_parameters()

      self.emb_dim = emb_dim
      self.nhead = nhead

      self.stride = stride
      self.k = k

      self.pad_vec = nn.Parameter(torch.randn(emb_dim))
      self.left_vec = nn.Parameter(torch.randn(emb_dim,1))
      self.right_vec = nn.Parameter(torch.randn(emb_dim,1))

   def reset_parameters(self):
      for p in self.parameters():
         if p.dim() > 1:
            nn.init.xavier_normal_(p)

   def forward(self, x, pos):
      # x.shape cca. N x l x 300
      b, l, emb = x.shape
      k, s = self.k, self.stride
      
      nl = math.ceil((l-k)/s)
      l_ = nl * s + k
      
      t = torch.zeros(size=(b, l_, emb))

      t[:, :l, :] = x
      t[:, l:, :] = torch.tile(self.pad_vec, dims=(l_-l, 1))
      while nl >= 1:
         c = torch.zeros(size=(b, nl, emb))
         for i in range(nl):
            c[:, i, :] = self.encoder(t[:, i*s:i*s+k, :], pos).reshape(-1, emb)
         t = c
         l = c.shape[1]
         nl = math.ceil((l-k)/s)
         l_ = nl * s + k

      return t

class TransformerEncoder(nn.Module):

   def __init__(self,
                emb_dim,
                nhead,
                num_layers,
                k,
                *args,
                dropout=0.1,
                **kwargs):
      super(TransformerEncoder, self).__init__(*args, **kwargs)
      self.layers = nn.ModuleList(TELayer(emb_dim, nhead, dropout=dropout) for _ in range(num_layers))
      self.linear = nn.Linear(k, 1)
      self.num_layers = num_layers

   def forward(self, x, pos):
      y = x
      for layer in self.layers:
         y = layer(y, pos)

      y = y.permute(0,2,1)
      y = self.linear(y)
      y = y.permute(0,2,1)
      return y

class TELayer(nn.Module):
   
   def __init__(self,
                emb_dim,
                nhead, 
                *args,
               dim_forward=512,
               dropout=0.1,
               **kwargs):
      super().__init__(*args, **kwargs)

      emb_dim = emb_dim
      nhead = nhead

      self.attn = nn.MultiheadAttention(emb_dim, nhead, dropout=dropout, batch_first=True)

      self.linear1 = nn.Linear(emb_dim, dim_forward)
      self.dropout = nn.Dropout(dropout)
      self.linear2 = nn.Linear(dim_forward, emb_dim)
      
      self.norm1 = nn.LayerNorm(emb_dim)
      self.norm2 = nn.LayerNorm(emb_dim)

      self.dropout1 = nn.Dropout(dropout)
      self.dropout2 = nn.Dropout(dropout)

      self.activation = nn.ReLU()

   def add_pos_embed(self, x, pos):
      return x if pos is None else pos(x)

   def forward(self, x, pos):
      q = k = self.add_pos_embed(x, pos)

      x2, att = self.attn(query=q,
                     key=k,
                     value=x)
      x = x + self.dropout1(x2)
      x = self.norm1(x)

      x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
      x = x + self.dropout2(x2)
      x = self.norm2(x)
      return x 


if __name__ == '__main__':
   # Test with pos emb
   N, l, emb = 50, 41, 300
   X = torch.randn(size=(N, l, emb))
   
   k = 6
   stride = 4
   emb_dim = 300

   model = RTransformer(k=k, stride=stride, emb_dim=emb_dim)
   from pos_emb import PositionalEmbedding
   pos = PositionalEmbedding(length=k, emb=emb_dim)

   xp = model.forward(X, pos)
   assert xp.shape == (50, 1, 300)