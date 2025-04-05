from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

def create_preprocessor():
    """Create preprocessing pipeline"""
    categorical_features = ['airline', 'source_city', 'departure_time', 
                          'stops', 'arrival_time', 'destination_city', 'class']
    numerical_features = ['duration', 'days_left']
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def save_preprocessor(preprocessor, output_path):
    """Save preprocessing pipeline"""
    joblib.dump(preprocessor, output_path)

if __name__ == "__main__":
    preprocessor = create_preprocessor()
    save_preprocessor(preprocessor, '../models/preprocessor.pkl')