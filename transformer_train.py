import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from data_loader import PARLAMENTS
from dataset import ParlaDataset, pad_collate_fn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter

class TransformerModel(nn.Module):
    def __init__(self, input_size=300, hidden_dim=300, num_layers=2, nhead=4, dropout=0.1, device='cpu'):
        super(TransformerModel, self).__init__()
        self.device = device
        self.input_size = input_size
        self.embedding = nn.Linear(input_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.act = nn.ReLU()
    
    def forward(self, x):
        #print(f'Initial shape: {x.shape}')
        x = x.float()
        x = self.embedding(x)
        #print(f'After embedding: {x.shape}')
        x = self.positional_encoding(x)
        #print(f'After positional encoding: {x.shape}')
        x = x.transpose(0, 1) 
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1) 
        #print(f'After transformer encoder: {x.shape}')
        x = x.mean(dim=1) 
        #print(f'After mean pooling: {x.shape}')
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        #print(f'After fully connected layers: {x.shape}')
        return x
    
    def predict(self, x):
        with torch.no_grad():
            out = self.forward(x)
            out = torch.sigmoid(out)
            out = torch.round(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds maximum positional encoding length {self.pe.size(1)}")
        #print(f'Positional Encoding Shape: {self.pe[:, :seq_len, :].shape}')
        #print(f'Input Shape: {x.shape}')
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return self.dropout(x)

for parlament in PARLAMENTS:
   for preprocess in [True, False]:
        print(f'Using model on "{parlament}" parlament with preprocessing={preprocess}')
      
        ds_train = ParlaDataset(parlament=parlament, set='train', preprocess=preprocess)
        ds_valid = ParlaDataset(parlament=parlament, set='valid', preprocess=preprocess)
        ds_test = ParlaDataset(parlament=parlament, set='test', preprocess=preprocess)

        shuffle = True
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = 4  
        lr = 1e-4
        epoch = 10

        emb_dim = 300
        num_layers = 2 
        hidden_dim = 300  
        nhead = 4
        dropout = 0.1
        gradient_clip = 1

        model = TransformerModel(input_size=emb_dim, hidden_dim=hidden_dim, num_layers=num_layers, nhead=nhead, dropout=dropout, device=device)
        model = model.to(device)

        name = 'transformer'
        if preprocess:
            name = 'preprocess'+name

        print(f'Model consists of {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters')

        trainloader = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate_fn)
        validloader = DataLoader(dataset=ds_valid, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate_fn)
        testloader = DataLoader(dataset=ds_test, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate_fn)

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

                out = model(text)
                
                loss = criterion(out, label) 
                loss.backward()
                #nn.utils.clip_grad_value_(model.parameters(), gradient_clip)
                optim.step()
                pbar.set_description(f'Epoch {epo} Loss: {loss.item()}')
                writer.add_scalar('Loss', loss.item(), epo*len(trainloader) + itern)

            acc = []
            prec = []
            rec = []
            f1 = []
            losses = []
            for text, label in tqdm(validloader):
                model.eval()
                
                text = text.to(device)
                label = label.to(device, torch.float32)

                loss = criterion(model(text), label)
                losses.append(loss.item())
                y_ = model.predict(text)

                Yt, Yp = label.cpu().detach().numpy(), y_.cpu().detach().numpy().reshape(-1, 1)
                Yt, Yp = Yt.flatten(), Yp.flatten()

                accuracy = accuracy_score(Yt, Yp)
                precision, recall, f1score, _ = precision_recall_fscore_support(Yt, Yp, average='binary', zero_division=1.0)

                acc.append(accuracy)
                prec.append(precision)
                rec.append(recall)
                f1.append(f1score)
            
            writer.add_scalar('Loss avg. valid', sum(losses)/len(losses), epo*len(trainloader) + itern)
            print(f'Avg. accuracy on validation is {sum(acc)/len(acc):.2f}')
            print(f'Avg. precision on validation is {sum(prec)/len(prec):.2f}')
            print(f'Avg. recall on validation is {sum(rec)/len(rec):.2f}')
            print(f'Avg. f1 on validation is {sum(f1)/len(f1):.2f}')

        acc = []
        prec = []
        rec = []
        f1 = []
        for text, label in tqdm(testloader):
            model.eval()

            text = text.to(device)
            label = label.to(device, torch.float32)

            y_ = model.predict(text)

            Yt, Yp = label.cpu().detach().numpy(), y_.cpu().detach().numpy().reshape(-1, 1)
            Yt, Yp = Yt.flatten(), Yp.flatten()

            accuracy = accuracy_score(Yt, Yp)
            precision, recall, f1score, _ = precision_recall_fscore_support(Yt, Yp, average='binary', zero_division=1.0)

            acc.append(accuracy)
            prec.append(precision)
            rec.append(recall)
            f1.append(f1score)

        print(f'Avg. accuracy on test is {sum(acc)/len(acc):.2f}')
        print(f'Avg. precision on test is {sum(prec)/len(prec):.2f}')
        print(f'Avg. recall on test is {sum(rec)/len(rec):.2f}')
        print(f'Avg. f1 on test is {sum(f1)/len(f1):.2f}')

        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optim.state_dict()}
        torch.save(state, f'runs/{name}_model.pth')

        with open(f'runs/{name}_par_{parlament}_results', 'w') as f:
            f.write(f'Avg. accuracy on test is {sum(acc)/len(acc):.2f}\n')
            f.write(f'Avg. precision on test is {sum(prec)/len(prec):.2f}\n')
            f.write(f'Avg. recall on test is {sum(rec)/len(rec):.2f}\n')
            f.write(f'Avg. f1 on test is {sum(f1)/len(f1):.2f}\n')
