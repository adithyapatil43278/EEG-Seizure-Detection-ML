# 🧠 EEG-Based Epileptic Seizure Detection using Neural Networks

## Overview

This project implements a machine learning pipeline to classify EEG (Electroencephalogram) signals from the Bangalore EEG dataset into different neurological states, including seizure and non-seizure conditions. Two models were developed and compared — a baseline Logistic Regression model and a Deep Neural Network (DNN) with extensive regularization and normalization layers.

The goal was to explore whether neural architectures could significantly outperform classical methods in EEG signal classification.

---

## ⚙️ Methodology

### Data Preprocessing

```python
# Importing the dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
EEG_data = pd.read_csv('BEED_Data[1].csv')

# Split features and labels
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Baseline Model – Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, log_reg.predict(X_test))
print(f"Accuracy: {accuracy}")
```

Achieved moderate accuracy (~47%), limited by non-linearity in EEG data.

### Advanced Model – Deep Neural Network

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2

# Build the model
model = Sequential()
model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01), input_shape=(input_dim,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
# Add more layers as needed

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, callbacks=[early_stopping])
```

Constructed a 22-layer fully connected neural network with:

- Dense layers ranging from 512 → 16 neurons.
- Batch Normalization after every layer.
- Leaky ReLU and ReLU activations.
- L2 regularization to reduce overfitting.
- Dropout (0.2) for better generalization.
- Early stopping used to monitor validation loss.

### Evaluation Metrics

- Precision, recall, F1-score, accuracy.
- Confusion matrices plotted for visual interpretation.

---

## 📊 Results

| Model                 | Accuracy | F1-Score (avg) | Remarks                          |
|-----------------------|----------|----------------|----------------------------------|
| Logistic Regression   | 47.4%   | 0.48           | Poor separation between classes  |
| Deep Neural Network   | 92.8%   | 0.93           | Excellent performance; clear class boundaries |

### Neural Network Confusion Matrix

- Demonstrated balanced precision across all four EEG classes.
- Loss curve showed stable convergence without overfitting.

---

## 🧩 Technologies Used

| Category          | Libraries                  |
|-------------------|----------------------------|
| Data Processing   | pandas, numpy, scikit-learn |
| Visualization     | matplotlib, seaborn        |
| Deep Learning     | tensorflow, keras          |

---

## 📁 Repository Structure

```
EEG-Seizure-Detection/
│
├── EEG_Bangalore.ipynb        # Google Colab notebook
├── Project_Report.pdf          # Final project documentation
├── Project_Report.docx         # Editable version
├── requirements.txt            # Dependencies (TensorFlow, sklearn, etc.)
└── README.md                   # This file
```

---

## 🔍 How to Run

1. Clone this repository:

```bash
git clone https://github.com/<your-username>/EEG-Seizure-Detection.git
cd EEG-Seizure-Detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook:

```bash
jupyter notebook EEG_Bangalore.ipynb
```

(Or run directly on Google Colab.)

---

## 💡 Future Improvements

- Experiment with CNNs or RNNs for temporal feature extraction.
- Perform hyperparameter tuning via Bayesian optimization.
- Use real-time EEG streaming data for inference.

---

## 👤 Author

**Adithya Patil**  
B.Tech in Computer Science  
Passionate not only for Machine Learning but for Everything.