import torch
import numpy as np

from tqdm import tqdm

from dataset import ParlaDataset, pad_collate_fn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from model import RTransformer
from pos_emb import PositionalEmbedding

ds_train = ParlaDataset(parlament='at', set='train')
ds_valid = ParlaDataset(parlament='at', set='valid')

shuffle = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
lr = 1e-2
epoch = 5

emb_dim = 300
nhead = 2
num_layers = 2
k = 10
stride = 10
r = 10

# loading model and pos embedding
model = RTransformer(emb_dim=emb_dim,
                     nhead=nhead,
                     num_layers=num_layers,
                     k=k,
                     stride=stride,
                     r=r,
                     device=device)
model = model.to(device)
pos = PositionalEmbedding(length=k, emb=emb_dim, device=device)
pos = pos.to(device)

trainloader = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate_fn)
validloader = DataLoader(dataset=ds_valid, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate_fn)

optim = torch.optim.Adam(model.parameters(), lr=lr)

for epo in range(epoch):

   pbar = tqdm(trainloader)
   for text, label in pbar:
      model.train()
      model.zero_grad()

      text = text.to(device)
      label = label.to(device)

      loss = model.get_loss(text, pos, label)
      loss.backward(retain_graph=True)
      optim.step()

      pbar.set_description(f'Loss: {loss.item()}')

   acc = []
   prec = []
   rec = []
   f1 = []
   for text, label in tqdm(validloader):
      model.eval()
      
      text = text.to(device)
      label = label.to(device)

      y_ = model.predict(text, pos)

      Yt, Yp = label.cpu().detach().numpy(), y_.cpu().detach().numpy().reshape(-1, 1)
      Yt, Yp = Yt.flatten(), Yp.flatten()

      accuracy = accuracy_score(Yt, Yp)
      precision, recall, f1score, _ = precision_recall_fscore_support(Yt, Yp, average='binary')

      acc.append(accuracy)
      prec.append(precision)
      rec.append(recall)
      f1.append(f1score)
   
   print(f'Avg. accuracy on validation is {sum(acc)/len(acc):.2f}')
   print(f'Avg. precision on validation is {sum(prec)/len(prec):.2f}')
   print(f'Avg. recall on validation is {sum(rec)/len(rec):.2f}')
   print(f'Avg. f1 on validation is {sum(f1)/len(f1):.2f}')

