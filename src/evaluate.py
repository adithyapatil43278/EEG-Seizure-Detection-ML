# src/evaluate.py

import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def evaluate_lg(x_test, y_test):
    model = joblib.load("models/lg_model.pkl")
    y_pred_lgr = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred_lgr)
    print("Logistic Regression Accuracy:", acc)

    print(classification_report(y_test, y_pred_lgr))

    cm = confusion_matrix(y_test, y_pred_lgr)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.show()


def evaluate_nn(x_test, y_test):
    nn_model = tf.keras.models.load_model("models/nn_model.h5")
    y_pred_probab = nn_model.predict(x_test)
    y_pred_nn = np.argmax(y_pred_probab, axis=1)

    acc = accuracy_score(y_test, y_pred_nn)
    print("Neural Network Accuracy:", acc)

    print(classification_report(y_test, y_pred_nn))

    c = confusion_matrix(y_test, y_pred_nn)
    plt.figure(figsize=(6,4))
    sns.heatmap(c, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Neural Network")
    plt.show()