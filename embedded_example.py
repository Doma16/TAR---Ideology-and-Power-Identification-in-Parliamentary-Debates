
if __name__ == '__main__':
   import pandas as pd

   from podium import Vocab, Field, LabelField
   from podium.datasets import TabularDataset
   from podium.vectorizers import GloVe

   # hyperparameter
   fixed_length = 1000

   vocab = Vocab(min_freq=2)
   S = Field(name='text',
             numericalizer=vocab,
             fixed_length=fixed_length,
             pretokenize_hooks=str.lower)
   L = LabelField('label')
   fields = {
        'text': S,
        'label': L
   }

   df = pd.read_csv('data/orientation_data.tsv')

   train = TabularDataset.from_pandas(df, fields)
   train.finalize_fields()

   glove = GloVe()
   embeddings = glove.load_vocab(vocab)

   train_batch = train.batch(add_padding=True)

   # samo prvih 100 ogromne kolicine podataka
   embedded = embeddings[train_batch.text[:100]]
   assert embedded.shape == (100, fixed_length, 300)
   breakpoint()