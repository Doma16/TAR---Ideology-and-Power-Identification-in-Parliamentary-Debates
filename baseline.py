import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score

def logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def svm(X_train, y_train):
    model = SVC()
    model.fit(X_train, y_train)
    return model

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy, conf_matrix, f1, recall, precision

if __name__ == '__main__':
    
    #X (Nxlx300)
    #y array of elements 0 or 1
    N, l, features = X.shape

    #reshaping X: (N, l, 300) to (N, l*300)
    X = X.reshape(N, -1)

    #spliting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #training
    logistic_model = logistic_regression(X_train, y_train)
    svm_model = svm(X_train, y_train)

    #predictions
    y_pred_log = logistic_model.predict(X_test)
    y_pred_svm = svm_model.predict(X_test)

    #evaluatin
    log_acc, log_conf, log_f1, log_recall, log_prec = evaluate_model(y_test, y_pred_log)
    print("Logistic Regression:")
    print("Accuracy:", log_acc)
    print("Confusion Matrix:\n", log_conf)
    print("F1 Score:", log_f1)
    print("Recall:", log_recall)
    print("Precision:", log_prec)

    svm_acc, svm_conf, svm_f1, svm_recall, svm_prec = evaluate_model(y_test, y_pred_svm)
    print("\nSVM:")
    print("Accuracy:", svm_acc)
    print("Confusion Matrix:\n", svm_conf)
    print("F1 Score:", svm_f1)
    print("Recall:", svm_recall)
    print("Precision:", svm_prec)
