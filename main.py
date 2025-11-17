# main.py

from src.preprocess import preprocess
from src.train import train_lg, train_nn
from src.evaluate import evaluate_lg, evaluate_nn

# Step 1: Load and preprocess data
X_train, X_test, y_train, y_test = preprocess("Data/BEED_Data.csv")

# Step 2: Train models
train_lg(X_train, y_train)
train_nn(X_train, y_train)

# Step 3: Evaluate models
evaluate_lg(X_test, y_test)
evaluate_nn(X_test, y_test)