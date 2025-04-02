import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Load the model and scaler with error handling
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error loading scaler: {str(e)}")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

# Add a cache decorator to improve performance
@st.cache_data
def predict_quality(input_data):
    input_scaled = scaler.transform(input_data)
    return model.predict(input_scaled)[0]

# Title and description
st.title("🍷 Wine Quality Predictor")
st.write("Enter wine properties to predict its quality")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", min_value=4.0, max_value=16.0, value=7.0, step=0.1)
    volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
    citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=16.0, value=2.0, step=0.1)
    chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0, max_value=100, value=30, step=1)

with col2:
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0, max_value=300, value=100, step=1)
    density = st.number_input("Density", min_value=0.9, max_value=1.1, value=1.0, step=0.001)
    ph = st.number_input("pH", min_value=2.0, max_value=4.0, value=3.0, step=0.1)
    sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
    alcohol = st.number_input("Alcohol", min_value=8.0, max_value=15.0, value=10.0, step=0.1)

# Create a button to make prediction
if st.button("Predict Quality", type="primary"):
    try:
        # Create input array
        input_data = np.array([[
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
            ph, sulphates, alcohol
        ]])
        
        # Make prediction using cached function
        prediction = predict_quality(input_data)
        
        # Display result
        st.write("### Prediction Result")
        quality_interpretation = {
            3: "Poor",
            4: "Fair",
            5: "Average",
            6: "Good",
            7: "Very Good",
            8: "Excellent"
        }
        
        st.success(f"Predicted Quality: {prediction}/8 ({quality_interpretation.get(prediction, 'Unknown')})")
        
        # Show feature importance
        st.write("### Most Important Factors")
        feature_importance = pd.DataFrame({
            'Feature': ['Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar',
                       'Chlorides', 'Free Sulfur Dioxide', 'Total Sulfur Dioxide', 'Density',
                       'pH', 'Sulphates', 'Alcohol'],
            'Importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False).head(5)
        st.bar_chart(feature_importance.set_index('Feature'))
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}") 