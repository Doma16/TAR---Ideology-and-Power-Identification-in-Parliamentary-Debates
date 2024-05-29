from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.linear_model import SGDClassifier
import numpy as np


def calculate_f1(model, data):
    y_true = []
    y_pred = []
    for embeddings, labels in data:
        embeddings_2d = embeddings.reshape(embeddings.shape[0], -1)
        batch_pred = model.predict(embeddings_2d)
        y_true.extend(labels.reshape(-1))
        y_pred.extend(batch_pred)
    
    #f1 = f1_score(y_true, y_pred, average='binary')
    prec, recall, fscore, supp = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return prec, recall, fscore, supp

def train_model(data):
    model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)

    for embeddings, labels in data:
        embeddings_2d = embeddings.reshape(embeddings.shape[0], -1)
        labels_1d = labels.reshape(-1)
        model.partial_fit(embeddings_2d, labels_1d, classes=np.unique(labels_1d))

    return model


if __name__ == '__main__':
    from data_loader import DataLoader
    import time 
    start = time.time()
    data = DataLoader(parlament='at', batch_size=32, padding=True)
    trained_model= train_model(data)
    
    data2 = DataLoader(parlament='at', set='valid', batch_size=32, padding=True)
    prec, recall, f1, _ = calculate_f1(trained_model, data2)
    print(f'precision: {prec:.2f} recall: {recall:.2f} f1score: {f1:.2f}')
    end = time.time()
    print(f'Time: {(end-start):.3f}s')
