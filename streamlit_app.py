#!/usr/bin/env python3
# streamlit_app.py - minimal UI template (replace with your model file paths)

import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title='Sleep Disorder Risk Calculator', layout='centered')

st.title('Sleep Disorder Risk Calculator (Demo)')

# Try to load model and scaler if present
MODEL_PATH = 'models/sleep_risk_model.pkl'
SCALER_PATH = 'models/feature_scaler.pkl'
FEATURES_PATH = 'models/feature_names.pkl'

model = None
scaler = None
feature_names = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning(f'Could not load model: {e}')
if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        st.warning(f'Could not load scaler: {e}')
if os.path.exists(FEATURES_PATH):
    try:
        feature_names = joblib.load(FEATURES_PATH)
    except Exception as e:
        st.warning(f'Could not load feature names: {e}')

# Simple inputs (match feature order in your model)
age = st.number_input('Age', min_value=18, max_value=100, value=40)
sex = st.selectbox('Sex', ['Male','Female'])
bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0)
exercise = st.slider('Exercise (min/week)', 0, 500, 150)
calories = st.number_input('Calories per day', min_value=0, max_value=10000, value=2000)
fiber = st.number_input('Fiber (g/day)', min_value=0.0, max_value=200.0, value=15.0)
added_sugar = st.number_input('Added sugar (g/day)', min_value=0.0, max_value=1000.0, value=30.0)
caffeine = st.number_input('Caffeine (mg/day)', min_value=0, max_value=2000, value=200)
alcohol = st.number_input('Alcohol drinks/week', min_value=0.0, max_value=100.0, value=3.0)
smoker = st.selectbox('Current smoker', ['No','Yes'])
depression = st.number_input('Depression score (PHQ-9)', min_value=0, max_value=27, value=3)
systolic = st.number_input('Systolic BP', min_value=70, max_value=250, value=120)
diastolic = st.number_input('Diastolic BP', min_value=40, max_value=150, value=80)

if st.button('Calculate Risk'):\n    X = np.array([[age, 1 if sex=='Male' else 2, bmi, exercise, calories, fiber, added_sugar, caffeine, alcohol, 1 if smoker=='Yes' else 0, depression, systolic, diastolic]])\n    if scaler is not None:\n        Xs = scaler.transform(X)\n    else:\n        Xs = X\n    if model is not None:\n        try:\n            risk = model.predict_proba(Xs)[0,1]\n            st.metric('Sleep Disorder Risk', f\"{risk*100:.1f}%\")\n        except Exception as e:\n            st.error(f'Error making prediction: {e}')\n    else:\n        st.info('No model found. Use this UI after training and saving model to models/sleep_risk_model.pkl')\n