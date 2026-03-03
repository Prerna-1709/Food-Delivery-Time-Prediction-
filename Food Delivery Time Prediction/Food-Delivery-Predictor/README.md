# 🍕 Food-Delivery-Predictor

A machine learning project to predict food delivery times.

## Project Structure

```
Food-Delivery-Predictor/
├── data/
│   └── raw/                  ← Place Food_Delivery_Time_Prediction.csv here
├── notebooks/                ← Jupyter notebooks for EDA
├── src/
│   ├── preprocessing/        ← Feature engineering & data cleaning
│   ├── models/               ← Model training & evaluation
│   └── utils/
│       └── data_loader.py    ← Dataset loading & sanity checks
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create and activate the virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt
```

## Running the Data Loader

```bash
python src/utils/data_loader.py
```

> **Note:** Place `Food_Delivery_Time_Prediction.csv` inside `data/raw/` before running.
