import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path

# Get the absolute path to the directory containing this script
current_dir = Path(__file__).parent

# Construct absolute paths to model files
MODEL_PATH = current_dir.parent / "models" / "best_model.pkl"
PREPROCESSOR_PATH = current_dir.parent / "models" / "preprocessor.pkl"

# Load model and preprocessor


@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


model, preprocessor = load_model()

# App UI
st.title('✈️ Airfare Price Prediction')

with st.form('prediction_form'):
    st.header('Enter Flight Details')

    col1, col2 = st.columns(2)

    with col1:
        airline = st.selectbox(
            'Airline', ['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo', 'Air_India'])
        source_city = st.selectbox('Departure City', [
                                   'Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
        departure_time = st.selectbox('Departure Time', [
                                      'Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'])
        stops = st.selectbox('Number of Stops', ['zero', 'one', 'two_or_more'])

    with col2:
        arrival_time = st.selectbox('Arrival Time', [
                                    'Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'])
        destination_city = st.selectbox('Destination City', [
                                        'Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
        flight_class = st.selectbox('Class', ['Economy', 'Business'])
        duration = st.number_input(
            'Duration (hours)', min_value=1.0, max_value=30.0, value=2.5, step=0.1)
        days_left = st.number_input(
            'Days Until Departure', min_value=1, max_value=50, value=7)

    submitted = st.form_submit_button('Predict Price')

    if submitted:
        try:
            # Create input DataFrame
            input_data = pd.DataFrame({
                'airline': [airline],
                'source_city': [source_city],
                'departure_time': [departure_time],
                'stops': [stops],
                'arrival_time': [arrival_time],
                'destination_city': [destination_city],
                'class': [flight_class],
                'duration': [duration],
                'days_left': [days_left]
            })

            # Preprocess input
            processed_input = preprocessor.transform(input_data)

            # Make prediction
            prediction = model.predict(processed_input)

            st.success(f'### Predicted Price: ₹{prediction[0]:.2f}')
            st.balloons()
        except Exception as e:
            st.error(f"Error making prediction: {e}")
