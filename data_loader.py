import random
import pandas as pd
from podium import Vocab, Field, LabelField
from podium.datasets import TabularDataset
from podium.vectorizers import GloVe
import numpy as np
import os
import time



PARLAMENTS = {
    'at':'orientation-at-train.tsv',
    'ba':'orientation-ba-train.tsv',
    'be':'orientation-be-train.tsv',
    'bg':'orientation-bg-train.tsv',
    'cz':'orientation-cz-train.tsv',
    'dk':'orientation-dk-train.tsv',
    'ee':'orientation-ee-train.tsv',
    #'es-ct':'orientation-es-ct-train.tsv', #drop for speed
    #'es-ga':'orientation-es-ga-train.tsv', #drop for speed
    'es':'orientation-es-train.tsv',
    'fi':'orientation-fr-train.tsv',
    'gb':'orientation-gb-train.tsv',
    'hr':'orientation-gr-train.tsv',
    'hu':'orientation-hr-train.tsv',
    'is':'orientation-hu-train.tsv',
    'it':'orientation-it-train.tsv',
    'lv':'orientation-lv-train.tsv',
    'nl':'orientation-nl-train.tsv',
    #'no':'orientation-no-train.tsv',
    #'pl':'orientation-pl-train.tsv',
    #'pt':'orientation-pt-train.tsv',
    #'rs':'orientation-rs-train.tsv',
    #'se':'orientation-se-train.tsv',
    #'si':'orientation-si-train.tsv',
    #'tr':'orientation-tr-train.tsv',
    #'ua':'orientation-ua-train.tsv',
}

class DataLoader:
    def __init__(self, parlament='at', set='train', batch_size=1, shuffle=True, padding=False, preprocess=False):
        set = set.lower()
        assert set in ['train', 'valid', 'test']
        self.set = set


        data_file = PARLAMENTS[parlament]
        if preprocess:
            data_file='stopword'+data_file

        self.file_path = os.path.join('data', 'out', data_file)
        self.parlament = parlament
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = None
        self.current_index = 0
        self.num_batches = None
        self.embeddings = None
        self.train = None
        self.dataset = pd.read_csv(self.file_path, sep='\t')
        self.padding = padding

        self.build_vocab()

        if self.shuffle:
            random.shuffle(self.indices)

    def build_vocab(self):
        fixed_length = 1000

        self.vocab = Vocab(min_freq=2)
        S = Field(name='text',
                  numericalizer=self.vocab,
                  fixed_length=fixed_length,
                  pretokenize_hooks=str.lower)
        L = LabelField('label')
        fields = {
                'text': S,
                'label': L
        }
        self.fields = fields

        train = TabularDataset.from_pandas(pd.read_csv(self.file_path, sep='\t'), fields)
        train.finalize_fields()
        self.train_batch = train.batch(add_padding=self.padding)

        self.train = train
        self.indices = list(range(len(train)))

        percent60 = round(len(self.indices) * 0.60)
        
        self.indices = self.indices[:percent60] if set == 'train' else self.indices[percent60:]
        if set != 'train':
            self.indices = self.indices[:len(self.indices)//2] if set == 'valid' else self.indices[len(self.indices)//2:]
        self.num_batches = int(np.ceil(len(self.indices) / self.batch_size))

        glove = GloVe()
        self.embeddings = glove.load_vocab(self.vocab)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return self
    
    #used for pytorch dataset
    def __getitem__(self, idx):
        
        index = self.indices[idx]
        data = self.train_batch.text[index]
        label = self.train_batch.label[index]

        data = np.array([self.embeddings[i] for i in data])
        return (data, label)

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
    
        batch_data = [self.train_batch.text[i] for i in batch_indices]
        batch_label = [self.train_batch.label[i] for i in batch_indices]
    
        self.current_index += self.batch_size
        if self.padding:
            batch_embeddings = self.embeddings[batch_data]
        else:
            batch_embeddings = [ self.embeddings[i] for i in batch_data ]

        labels = np.hstack(batch_label)
        return batch_embeddings, labels
            