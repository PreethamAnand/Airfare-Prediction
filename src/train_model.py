import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def load_processed_data(data_dir):
    """Load processed data"""
    X_train = pd.read_pickle(os.path.join(data_dir, 'X_train.pkl'))
    X_test = pd.read_pickle(os.path.join(data_dir, 'X_test.pkl'))
    y_train = pd.read_pickle(os.path.join(data_dir, 'y_train.pkl'))
    y_test = pd.read_pickle(os.path.join(data_dir, 'y_test.pkl'))
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train, preprocessor):
    """Train and evaluate multiple models"""
    # Preprocess data
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    # Train models
    for name, model in models.items():
        model.fit(X_train_processed, y_train)
    
    return models

def evaluate_models(models, preprocessor, X_test, y_test):
    """Evaluate models and return metrics"""
    X_test_processed = preprocessor.transform(X_test)
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test_processed)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({'Model': name, 'MSE': mse, 'R2': r2})
    
    return pd.DataFrame(results)

def save_models(models, output_dir):
    """Save trained models"""
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, os.path.join(output_dir, f'{name.lower().replace(" ", "_")}.pkl'))

if __name__ == "__main__":
    import os
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data('../data/processed/')
    
    # Load preprocessor
    preprocessor = joblib.load('../models/preprocessor.pkl')
    
    # Train models
    models = train_models(X_train, y_train, preprocessor)
    
    # Evaluate
    results = evaluate_models(models, preprocessor, X_test, y_test)
    print(results)
    
    # Save models and results
    save_models(models, '../models/')
    results.to_csv('../models/model_performance.csv', index=False)
    
    # Save best model separately
    best_model_name = results.loc[results['R2'].idxmax(), 'Model']
    joblib.dump(models[best_model_name], '../models/best_model.pkl')