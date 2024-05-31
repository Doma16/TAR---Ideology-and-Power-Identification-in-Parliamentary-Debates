import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

from dataset import ParlaDataset, pad_collate_fn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from model import RTransformer
from pos_emb import PositionalEmbedding

parlament = 'ba'

ds_train = ParlaDataset(parlament=parlament, set='train')
ds_valid = ParlaDataset(parlament=parlament, set='valid')

shuffle = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
lr = 9e-5
epoch = 10

emb_dim = 300
nhead = 2
num_layers = 2
k = 5
stride = 5

# loading model and pos embedding
model = RTransformer(emb_dim=emb_dim,
                     nhead=nhead,
                     num_layers=num_layers,
                     k=k,
                     stride=stride,
                     device=device)
model = model.to(device)
pos = PositionalEmbedding(length=k, emb=emb_dim, device=device)
pos = pos.to(device)

print(f'Model consists of {model.count_parameters()} parameters')

trainloader = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate_fn)
validloader = DataLoader(dataset=ds_valid, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate_fn)

optim = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

for epo in range(epoch):

   pbar = tqdm(trainloader)
   for text, label in pbar:
      model.train()
      model.zero_grad()

      text = text.to(device)
      label = label.to(device, torch.float32)

      out = model.forward(text, pos)
      
      loss = criterion(out.flatten(1), label)
      loss.backward(retain_graph=True)
      optim.step()
      pbar.set_description(f'Epoch {epo} Loss: {loss.item()}')

   acc = []
   prec = []
   rec = []
   f1 = []
   for text, label in tqdm(validloader):
      model.eval()
      
      text = text.to(device)
      label = label.to(device)

      y_ = model.bce_predict(text, pos)

      Yt, Yp = label.cpu().detach().numpy(), y_.cpu().detach().numpy().reshape(-1, 1)
      Yt, Yp = Yt.flatten(), Yp.flatten()

      accuracy = accuracy_score(Yt, Yp)
      precision, recall, f1score, _ = precision_recall_fscore_support(Yt, Yp, average='macro', zero_division=1.0)

      acc.append(accuracy)
      prec.append(precision)
      rec.append(recall)
      f1.append(f1score)
   
   print(f'Avg. accuracy on validation is {sum(acc)/len(acc):.2f}')
   print(f'Avg. precision on validation is {sum(prec)/len(prec):.2f}')
   print(f'Avg. recall on validation is {sum(rec)/len(rec):.2f}')
   print(f'Avg. f1 on validation is {sum(f1)/len(f1):.2f}')

