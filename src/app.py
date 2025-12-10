import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model
import os

# Page Config
st.set_page_config(page_title='BAT OptiCure', layout='wide')

# Sidebar for Inputs
st.sidebar.header('Sensor Simulation')
temp = st.sidebar.slider('Barn Temperature (C)', 50, 90, 70, help='Optimal: 65-75C')
humid = st.sidebar.slider('Humidity (%)', 20, 100, 60, help='Optimal: 50-70%')
moisture = st.sidebar.slider('Leaf Moisture (%)', 10, 30, 18, help='Target: ~18%')

# Main Dashboard
st.title('OptiCure AI: Intelligent Curing Control')
st.markdown('System Status: ONLINE | Client: British American Tobacco')

st.info('This dashboard demonstrates MLOps in action. Changes pushed to GitHub trigger a retraining of the model below.')

# Load Model Logic
model_path = 'bat_curing_pipeline'

if os.path.exists(model_path + '.pkl'):
    try:
        model = load_model(model_path)
        st.sidebar.success('AI Model Loaded')

        # Prediction Logic
        input_df = pd.DataFrame({
            'barn_temperature': [temp],
            'barn_humidity': [humid],
            'leaf_moisture': [moisture],
            'airflow_rate': [3.5],
            'curing_time_hours': [72]
        })
        
        if st.button('Analyze Batch Quality', type='primary'):
            predictions = predict_model(model, data=input_df)
            grade = predictions['prediction_label'][0]
            score = predictions['prediction_score'][0]
            
            col1, col2 = st.columns(2)
            with col1:
                if grade == 'Premium':
                    st.success(f'PREDICTION: PREMIUM\n\nConfidence: {score:.2f}')
                else:
                    st.error(f'PREDICTION: STANDARD\n\nConfidence: {score:.2f}')
            with col2:
                st.metric('Temp Deviation', f'{temp - 70}C')
        
        # Show Metrics if available
        if os.path.exists('model_metrics.csv'):
            st.divider()
            with st.expander('View Model Performance (MLOps Logs)'):
                st.dataframe(pd.read_csv('model_metrics.csv'))
            
    except Exception as e:
        st.error(f'Error loading model: {e}')
else:
    st.warning('Model not found. Please run the training pipeline via GitHub Actions or locally.')
