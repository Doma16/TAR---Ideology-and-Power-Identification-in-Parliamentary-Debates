import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

from dataset import ParlaDataset, pad_collate_fn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from torch.utils.tensorboard import SummaryWriter

class ourRNN(nn.Module):
   def __init__(self, *args, 
                device='cpu',
                input_size=300,
                hidden_dim=300,
                num_layers=2,
                **kwargs):
      super().__init__(*args, **kwargs)
      self.rnn1 = nn.RNN(batch_first=True, device=device, input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True)
      
      self.act = nn.ReLU()
      self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
      self.fc2 = nn.Linear(hidden_dim, 1)

   def forward(self, x, h):
      out, hidden = self.rnn1(x, h)

      y = self.act(self.fc1(out[:, out.shape[1]//2, :]))
      y = self.fc2(y)
      return y
       
   def count_parameters(self):
      params = 0
      for param in self.parameters():
         params += param.numel()
      return params
   
   def predict(self, x, h):
      with torch.no_grad():
         out = self.forward(x, h)
         out = 1 / (1+torch.exp(-out))
         out = torch.round(out)
      return out

parlament = 'cz'
preprocess = True

ds_train = ParlaDataset(parlament=parlament, set='train', preprocess=preprocess)
ds_valid = ParlaDataset(parlament=parlament, set='valid', preprocess=preprocess)

shuffle = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
lr = 1e-5
epoch = 10

emb_dim = 300
num_layers = 4
hidden_dim = 600
gradient_clip = 1

# loading model and pos embedding
model = ourRNN(device=device, input_size=emb_dim, hidden_dim=hidden_dim, num_layers=num_layers)
model = model.to(device)

name = 'ourrnn'
if preprocess:
   name = 'preprocess'+name

print(f'Model consists of {model.count_parameters()} parameters')

trainloader = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate_fn)
validloader = DataLoader(dataset=ds_valid, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate_fn)

optim = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

writer = SummaryWriter(log_dir=f'runs/{name}_par_{parlament}')

for epo in range(epoch):

   pbar = tqdm(trainloader)
   for itern, (text, label) in enumerate(pbar):
      model.train()
      model.zero_grad()

      text = text.to(device)
      label = label.to(device, torch.float32)

      n,l,emb = text.shape
      h = torch.zeros(size=(num_layers*2, n, hidden_dim)).to(device)
      #hs = (torch.zeros(size=(num_layers, n, hidden_dim)).to(device) for _ in range(2))
      out = model.forward(text, h)
      
      loss = criterion(out, label)
      loss.backward(retain_graph=True)
      nn.utils.clip_grad_value_(model.parameters(), gradient_clip)
      optim.step()
      pbar.set_description(f'Epoch {epo} Loss: {loss.item()}')
      writer.add_scalar('Loss', loss.item(), epo*len(trainloader) + itern)

   acc = []
   prec = []
   rec = []
   f1 = []
   for text, label in tqdm(validloader):
      model.eval()
      
      text = text.to(device)
      label = label.to(device)

      n,l,emb = text.shape
      h = torch.zeros(size=(num_layers*2, n, hidden_dim)).to(device)
      #hs = (torch.zeros(size=(num_layers, n, hidden_dim)).to(device) for _ in range(2))
      y_ = model.predict(text, h)

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

#saving model
#NOTE you can overwrite existing model if you rerun script

state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optim.state_dict()}
torch.save(state, f'runs/{name}_model.pth')

# should get this results on test set !!!
with open(f'runs/{name}_par_{parlament}_results', 'w') as f:
   f.write(f'Avg. accuracy on validation is {sum(acc)/len(acc):.2f}\n')
   f.write(f'Avg. precision on validation is {sum(prec)/len(prec):.2f}\n')
   f.write(f'Avg. recall on validation is {sum(rec)/len(rec):.2f}\n')
   f.write(f'Avg. f1 on validation is {sum(f1)/len(f1):.2f}\n')