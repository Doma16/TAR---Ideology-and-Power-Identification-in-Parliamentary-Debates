import random
import pandas as pd
from podium import Vocab, Field, LabelField
from podium.datasets import TabularDataset
from podium.vectorizers import GloVe
import numpy as np

import time

class DataLoader:
    def __init__(self, file_path, batch_size, shuffle=True):
        
        self.file_path = file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = None
        self.current_index = 0
        self.num_batches = None
        self.embeddings = None
        self.train = None
        self.dataset = pd.read_csv(self.file_path, sep='\t')

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
        self.train_batch = train.batch(add_padding=True)

        self.train = train
        self.indices = list(range(len(train)))
        self.num_batches = int(np.ceil(len(train) / self.batch_size))

        glove = GloVe()
        self.embeddings = glove.load_vocab(self.vocab)


    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
    
        batch_data = [self.train_batch.text[i] for i in batch_indices]
        batch_label = [self.train_batch.label[i] for i in batch_indices]
    
        batch_embeddings = self.embeddings[batch_data]
        batch_labels = self.train_batch.label[batch_label]

        self.current_index += self.batch_size
        return batch_embeddings, batch_labels
