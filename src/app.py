import streamlit as st
import pandas as pd
import joblib
import os

# Page Config
st.set_page_config(page_title='BAT OptiCure', layout='wide')

# Sidebar for Inputs
st.sidebar.header('Sensor Simulation')
temp = st.sidebar.slider('Barn Temperature (C)', 50, 90, 70, help='Optimal: 65-75C')
humid = st.sidebar.slider('Humidity (%)', 20, 100, 60, help='Optimal: 50-70%')
moisture = st.sidebar.slider('Leaf Moisture (%)', 10, 30, 18, help='Target: ~18%')

# NEW: Add Model Version/Timestamp Display
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ MLOps Status")
if os.path.exists(model_path + '.pkl'):
    # Get the time the model file was last modified
    mod_time = os.path.getmtime(model_path + '.pkl')
    timestamp = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
    st.sidebar.success(f"Model Last Trained:\n{timestamp}")
else:
    st.sidebar.error("Model Missing")

# Main Dashboard
st.title('OptiCure AI: Intelligent Curing Control')
st.markdown('System Status: ONLINE | Client: British American Tobacco')

st.info('This dashboard demonstrates MLOps in action. Changes pushed to GitHub trigger a retraining of the model below.')

# Load Model Logic
model_path = 'bat_curing_pipeline.pkl'

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
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
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            
            col1, col2 = st.columns(2)
            with col1:
                grade = 'Premium' if prediction == 'Premium' else 'Standard'
                confidence = max(probability) * 100
                
                if grade == 'Premium':
                    st.success(f'PREDICTION: PREMIUM\n\nConfidence: {confidence:.1f}%')
                else:
                    st.error(f'PREDICTION: STANDARD\n\nConfidence: {confidence:.1f}%')
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
