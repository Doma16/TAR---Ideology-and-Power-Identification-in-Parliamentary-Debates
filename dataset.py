# this is dataset for pytorch
from data_loader import DataLoader as DL

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ParlaDataset(Dataset):
   def __init__(self, parlament='at', set='train', preprocess=False):
      super().__init__()

      self.dl = DL(parlament=parlament, set=set, padding=False, batch_size=1, shuffle=False, preprocess=preprocess)

   def __len__(self):
      return len(self.dl)
   
   def __getitem__(self, idx):
      data, label = self.dl[idx]
      return (data, label)

def pad_collate_fn(batch, pad_idx=0):
   texts, labels = zip(*batch)

   lengths = torch.tensor([text.shape[0] for text in texts])
   maxlen = max(lengths)

   out = torch.zeros(len(batch), maxlen, texts[0].shape[1])
   for idx, text in enumerate(texts):
      out[idx, :text.shape[0]] = torch.tensor(text)
   
   labels = torch.tensor(np.hstack(labels)).view(-1, 1)
   return out, labels


if __name__ == '__main__':

   pd = ParlaDataset()
   
   dpdl = DataLoader(pd, shuffle=True, batch_size=32, collate_fn=pad_collate_fn)

   texts, labels = next(iter(dpdl))
   
   print(texts.shape)
   print(labels.shape)