# src/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(path):
    df = pd.read_csv(path)

    x = df.drop(columns = "y")
    y = df["y"]

    scaler = StandardScaler()
    x_train_unscaled, x_test_unscaled, y_train, y_test = train_test_split(x, y, test_size=0.2)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_unscaled)
    x_test = scaler.transform(x_test_unscaled)
    return x_train, x_test, y_train, y_test