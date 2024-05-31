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
               device='cpu',
               **kwargs):
      super(RTransformer, self).__init__(*args, **kwargs)
      self.encoder = TransformerEncoder(emb_dim, nhead, num_layers, k, dropout=0, device=device)
      '''
      self.encoder = nn.Sequential(
         nn.Linear(in_features=k*emb_dim, out_features=emb_dim),
         nn.ReLU(),
         nn.Linear(emb_dim, emb_dim),
         nn.ReLU(),
         nn.Linear(emb_dim, emb_dim)
      )
      '''
      self.reset_parameters()

      self.emb_dim = emb_dim
      self.nhead = nhead
      self.device = device
      self.stride = stride
      self.k = k

      #self.pad_vec = nn.Parameter(torch.randn(emb_dim), requires_grad=True).to(device)
      self.left_vec = nn.Parameter(-torch.ones(emb_dim), requires_grad=True).to(device)
      self.right_vec = nn.Parameter(torch.ones(emb_dim), requires_grad=True).to(device)

      self.classify = nn.Linear(emb_dim, 1)

   def reset_parameters(self):
      for p in self.parameters():
         if p.dim() > 1:
            nn.init.xavier_normal_(p)

   def forward(self, x, pos=None):
      # x.shape cca. N x l x 300
      b, l, emb = x.shape
      k, s = self.k, self.stride
      
      nl = math.ceil(l/s)
      l_ = (nl-1) * s + k
      
      t = torch.zeros(size=(b, l_, emb)).to(self.device)

      t[:, :l, :] = x
      #t[:, l:, :] = torch.tile(self.pad_vec, dims=(l_-l, 1))
      while l > 1:
         c = torch.zeros(size=(b, nl, emb)).to(self.device)
         for i in range(nl):
            c[:, i, :] = self.encoder(t[:, i*s:i*s+k, :], pos).reshape(-1, emb) #.flatten(1,2))#
         l = c.shape[1]
         nl = math.ceil(l/s)
         l_ = (nl-1) * s + k

         t = torch.zeros(size=(b, l_, emb)).to(self.device)
         t[:, :l, :] = c

      out = self.classify(c)
      return out

   def get_loss(self, x, pos, label, criterion=nn.BCEWithLogitsLoss):
      out = self.forward(x, pos)
      
      sigmoid = 1 / (1 + torch.exp(-out)).flatten(1)

      criterion()
      breakpoint()
      '''
      maskl = label == 0
      maskr = label == 1

      out[maskl] -= self.left_vec
      out[maskr] -= self.right_vec

      out **= 2
      out = torch.sum(out, dim=(1,2))
      '''
      return torch.mean(out)
   
   def bce_predict(self, x, pos):
      with torch.no_grad():
         out = self.forward(x, pos)
         out = 1 / (1+torch.exp(-out))
         out = torch.round(out)
      return out.flatten(1)

   def predict(self, x, pos):
      out = self.forward(x, pos)

      leftD = torch.sum((out - self.left_vec)**2, dim=(1,2))
      rightD = torch.sum((out - self.right_vec)**2, dim=(1,2))

      diff = leftD - rightD
      
      diff[diff >= 0] = 1
      diff[diff < 0] = 0
      return diff
      
   def count_parameters(self):
      params = 0
      for param in self.parameters():
         params += param.numel()
      return params
class TransformerEncoder(nn.Module):

   def __init__(self,
                emb_dim,
                nhead,
                num_layers,
                k,
                *args,
                dropout=0.1,
                device='cpu',
                **kwargs):
      super(TransformerEncoder, self).__init__(*args, **kwargs)
      self.layers = nn.ModuleList(TELayer(emb_dim, nhead, dropout=dropout) for _ in range(num_layers)).to(device)
      self.linear = nn.Linear(k, 1).to(device)
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

   label = torch.zeros(50, 1)
   label[1] += 1

   model.get_loss(X, pos, label)
   model.predict(X, pos)