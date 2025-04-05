import pandas as pd
import numpy as np
from pathlib import Path
import pickle

def get_project_root():
    """Returns project root folder"""
    return Path(__file__).parent.parent

def load_data():
    """Load and preprocess raw data with proper path handling"""
    data_path = get_project_root() / 'data' / 'raw' / 'flight_prices.csv'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Convert duration to minutes
    df['duration'] = df['duration'].apply(lambda x: int(float(x)*60))
    
    # Drop duplicates
    return df.drop_duplicates()

def save_processed_data(df, output_path):
    """Robust saving of processed data"""
    try:
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with protocol=4 for better compatibility
        with open(output_path, 'wb') as f:
            pickle.dump(df, f, protocol=4)
            
        print(f"Successfully saved processed data to {output_path}")
        return True
    except Exception as e:
        print(f"Failed to save processed data: {str(e)}")
        # Clean up potentially corrupted file
        if output_path.exists():
            output_path.unlink()
        return False

if __name__ == "__main__":
    try:
        # Load and process data
        df = load_data()
        
        # Define output path
        output_path = get_project_root() / 'data' / 'processed' / 'processed_data.pkl'
        
        # Save processed data
        if save_processed_data(df, output_path):
            print("Data preprocessing completed successfully!")
        else:
            raise RuntimeError("Failed to save processed data")
            
    except Exception as e:
        print(f"Error in data preprocessing: {str(e)}")
        raise