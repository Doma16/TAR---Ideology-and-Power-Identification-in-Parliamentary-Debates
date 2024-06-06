from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.linear_model import SGDClassifier
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from data_loader import PARLAMENTS

def calculate_metrics(model, data):

    acc = []
    prec = []
    rec = []
    f1 = []
    for embeddings, labels in data:
        embeddings_2d = embeddings.reshape(embeddings.shape[0], -1)
        batch_pred = model.predict(embeddings_2d)
        
        y_true = labels.reshape(-1)
        y_pred = batch_pred
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=1.0)
        acc.append(accuracy)
        prec.append(precision)
        rec.append(recall)
        f1.append(fscore)

    return prec, rec, f1, acc

def train_model(data):
    model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)

    for embeddings, labels in data:
        embeddings_2d = embeddings.reshape(embeddings.shape[0], -1)
        labels_1d = labels.reshape(-1)
        model.partial_fit(embeddings_2d, labels_1d, classes=(0, 1))

    return model


if __name__ == '__main__':
    from data_loader import DataLoader
    import time 
    start = time.time()


    for parlament in PARLAMENTS:
       for preprocess in [True, False]:
            print(f'Using model on "{parlament}" parlament with preprocessing={preprocess}')
            name = 'svc'
            if preprocess:
                name = 'preprocess'+name

            data = DataLoader(parlament=parlament, batch_size=32, padding=True, preprocess=preprocess)
            trained_model= train_model(data)
            
            data2 = DataLoader(parlament=parlament, set='valid', batch_size=32, padding=True, preprocess=True)
            prec, rec, f1, acc = calculate_metrics(trained_model, data2)

            print(f'Avg. accuracy on validation is {sum(acc)/len(acc):.2f}')
            print(f'Avg. precision on validation is {sum(prec)/len(prec):.2f}')
            print(f'Avg. recall on validation is {sum(rec)/len(rec):.2f}')
            print(f'Avg. f1 on validation is {sum(f1)/len(f1):.2f}')

            data3 = DataLoader(parlament=parlament, set='test', batch_size=32, padding=True, preprocess=preprocess)
            prec, rec, f1, acc = calculate_metrics(trained_model, data3)

            print(f'Avg. accuracy on test is {sum(acc)/len(acc):.2f}')
            print(f'Avg. precision on test is {sum(prec)/len(prec):.2f}')
            print(f'Avg. recall on test is {sum(rec)/len(rec):.2f}')
            print(f'Avg. f1 on test is {sum(f1)/len(f1):.2f}')

            end = time.time()
            print(f'Time: {(end-start):.3f}s')
            with open(f'runs/{name}_par_{parlament}_results', 'w') as f:
                f.write(f'Avg. accuracy on test is {sum(acc)/len(acc):.2f}\n')
                f.write(f'Avg. precision on test is {sum(prec)/len(prec):.2f}\n')
                f.write(f'Avg. recall on test is {sum(rec)/len(rec):.2f}\n')
                f.write(f'Avg. f1 on test is {sum(f1)/len(f1):.2f}\n')