# 🧠 Food-Delivery-Intelligence

Advanced ML project for food delivery time prediction — regression, classification, deep learning, and clustering.

## Project Structure

```
Food-Delivery-Intelligence/
├── data/
│   ├── raw/                          ← Place Food_Delivery_Time_Prediction.csv here
│   └── processed/                    ← Auto-created on run
├── notebooks/                        ← EDA and experiment notebooks
├── src/
│   ├── preprocessing/                ← Feature engineering pipeline
│   ├── models/
│   │   ├── traditional/              ← Regression, Naive Bayes, KNN, Decision Tree
│   │   ├── deep_learning/            ← Neural networks (TensorFlow/Keras)
│   │   └── unsupervised/             ← Clustering (K-Means etc.)
│   ├── evaluation/                   ← Metrics, comparison utilities
│   └── utils/
│       └── config.py                 ← Global constants (paths, seeds, hyperparams)
├── models/saved/                     ← Persisted .joblib / .keras model files
├── reports/figures/                  ← Generated charts and plots
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt
```

## Key Config (`src/utils/config.py`)

| Constant | Value |
|---|---|
| `RANDOM_SEED` | `42` |
| `TEST_SIZE` | `0.20` |
| `DELAY_THRESHOLD_MIN` | `40` mins |
| `DL_EPOCHS` | `100` |
| `DL_BATCH_SIZE` | `32` |
