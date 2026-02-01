# Airfare Price Prediction

End-to-end machine learning project for predicting flight ticket prices using historical flight data. The project includes data preprocessing, feature engineering, model training/evaluation, and a Streamlit web app for interactive predictions.

## Features
- Cleaned and preprocessed dataset
- Multiple regression models (Linear Regression, Random Forest, XGBoost)
- Model evaluation metrics (MSE, R2)
- Streamlit UI for real-time price prediction

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Streamlit

## Project Structure
```
app/
  app.py                # Streamlit application

data/
  raw/                  # Raw dataset
  processed/            # Preprocessed data artifacts

models/
  best_model.pkl        # Best trained model
  preprocessor.pkl      # Feature preprocessing pipeline
  model_performance.csv # Evaluation metrics

notebooks/
  01_data_exploration.ipynb
  02_feature_engineering.ipynb
  03_model_training.ipynb

reports/
  figures/              # Charts/plots

src/
  data_preprocessing.py # Load/clean raw data
  feature_engineering.py# Build/save preprocessing pipeline
  train_model.py        # Train/evaluate models
  predict.py            # (reserved)

requirements.txt
```

## Setup
1. Create a virtual environment (optional but recommended).
2. Install dependencies:

```
pip install -r requirements.txt
```

## Run the Streamlit App
From the project root:

```
streamlit run app/app.py
```

## Training Pipeline (Optional)
If you want to reproduce model training:

1. Preprocess raw data:

```
python src/data_preprocessing.py
```

2. Create the preprocessing pipeline:

```
python src/feature_engineering.py
```

3. Train and evaluate models:

```
python src/train_model.py
```

> Note: The training script expects prepared train/test splits saved as `X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl` in `data/processed/`. These are produced in the notebooks.

## Outputs
- Best model saved to `models/best_model.pkl`
- Preprocessor saved to `models/preprocessor.pkl`
- Evaluation metrics saved to `models/model_performance.csv`

## Data
The dataset is located at `data/raw/flight_prices.csv`.

