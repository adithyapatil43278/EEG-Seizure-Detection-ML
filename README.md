# 🧠 EEG Seizure Classification (EEG‑ML)

Classifies epileptic seizure states from EEG tabular features. Early and reliable seizure detection supports faster diagnosis and clinical decision‑making.

---

## 📌 Problem & Motivation

- Goal: predict one of 4 EEG classes from 16 time/frequency features per sample.
- Why it matters: timely seizure detection can reduce diagnostic delays and improve patient outcomes by guiding treatment earlier.

---

## 📦 Data Source

- Dataset: BEED (tabular EEG features), stored at `Data/BEED_Data.csv`.
- Size: 8,000 rows × 17 columns → 16 features (`X1..X16`) + target `y` (4 classes: 0–3).
- Provenance: provided from the UCL Machine Learning repository.

---

## 🔧 Methodology

Pipeline implemented in `src/` and orchestrated by `main.py`.

- Preprocessing (`src/preprocess.py`)
	- Read CSV, split into features/label (`y`).
	- Standardize features with `StandardScaler`.
	- Hold‑out split: 80% train / 20% test.

- Baseline: Logistic Regression (`src/train.py::train_lg`)
	- `sklearn.linear_model.LogisticRegression` with default configuration.
	- Model persisted to `models/lg_model.pkl` via `joblib`.

- Neural Network (`src/train.py::train_nn`)
	- Keras Sequential MLP: Dense blocks 512→256→128→64→32→16, each with BatchNorm and `PReLU`, L2(0.001); Dropout(0.2) on deeper blocks.
	- Output: Dense(4, `softmax`).
	- Optimizer: Adam(lr=0.001); Loss: sparse categorical cross‑entropy.
	- EarlyStopping on `val_loss` (patience=10, restore best weights).
	- Trained up to 300 epochs, batch size 64, `validation_split=0.2`.
	- Saved to `models/nn_model.h5`.

- Evaluation (`src/evaluate.py`)
	- Reports accuracy, full classification report, and renders confusion matrices for both models.

---

## 📊 Results

Metrics below are also visible in the notebook (`notebook/EEG_Banglore_data.ipynb`).

### Summary

| Model                 | Accuracy |
|-----------------------|----------|
| Logistic Regression   | 0.4631   |
| Neural Network (MLP)  | 0.9688   |

### Neural Network: per‑class metrics

| Class | Precision | Recall | F1‑score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 1.00   | 1.00     | 422     |
| 1     | 0.94      | 0.99   | 0.96     | 354     |
| 2     | 0.96      | 0.96   | 0.96     | 421     |
| 3     | 0.97      | 0.93   | 0.95     | 403     |
|       |           |        |          |         |
| accuracy |        |        | 0.97     | 1600    |
| macro avg | 0.97  | 0.97   | 0.97     | 1600    |
| weighted avg | 0.97 | 0.97 | 0.97     | 1600    |

Notes:
- Test set size is 20% of the dataset (1,600 samples), consistent with preprocessing.
- Small run‑to‑run variation is possible due to randomness in splits/initialization.

---

## 🖼️ Visuals

- Loss curve (training vs validation) and confusion matrices are produced by the notebook and by `src/evaluate.py` during `main.py` execution.

---

## 🚀 Installation & Usage

Tested on Python 3.10+.

1) Clone the repo

```powershell
git clone https://github.com/adithyapatil43278/EEG-Seizure-Detection-ML.git
cd EEG-Seizure-Detection-ML
```

2) (Recommended) Create a virtual environment

```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
```

3) Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4) Run the pipeline

```powershell
python main.py
```

This will:
- Preprocess `Data/BEED_Data.csv`.
- Train Logistic Regression and Neural Network models.
- Evaluate both models and display classification reports and confusion matrices.

---

## 📂 Repository Structure

```
.
├── Data/
│   └── BEED_Data.csv
├── models/
├── notebook/
│   └── EEG_Banglore_data.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 🙏 Acknowledgements

- Dataset courtesy of the UCL Machine Learning Repository ((https://archive.ics.uci.edu/dataset/1134/beed:+bangalore+eeg+epilepsy+dataset)).
- Open‑source libraries: `numpy`, `pandas`, `scikit‑learn`, `tensorflow/keras`, `matplotlib`, `seaborn`, `joblib`, `h5py`.