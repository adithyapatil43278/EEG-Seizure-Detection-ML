# src/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(path):
    df = pd.read_csv(path)

    x = df.drop(columns = "y")
    y = df["y"]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2)
    return x_train, x_test, y_train, y_test