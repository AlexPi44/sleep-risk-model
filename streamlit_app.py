import streamlit as st
import joblib
import numpy as np
import os
import sys

st.set_page_config(page_title="Sleep Disorder Risk Calculator", layout="centered")

st.title("Sleep Disorder Risk Calculator")
st.markdown(
    "**Disclaimer:** This tool provides risk estimates, NOT a medical diagnosis. "
    "Consult a healthcare provider for sleep concerns."
)

# Model paths
MODEL_PATH = "models/sleep_risk_model.pkl"
SCALER_PATH = "models/feature_scaler.pkl"
FEATURES_PATH = "models/feature_names.pkl"

# Load model artifacts
model = None
scaler = None
feature_names = None
load_error = None

if not os.path.exists(MODEL_PATH):
    load_error = (
        f"Model not found at {MODEL_PATH}. "
        "Train the model first using: python src/merge_and_engineer.py && jupyter notebooks/01_data_prep.ipynb"
    )
elif not os.path.exists(SCALER_PATH):
    load_error = f"Scaler not found at {SCALER_PATH}"
elif not os.path.exists(FEATURES_PATH):
    load_error = f"Feature names not found at {FEATURES_PATH}"
else:
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURES_PATH)
    except Exception as e:
        load_error = f"Failed to load model artifacts: {e}"

if load_error:
    st.error(f"⚠️ {load_error}")
    st.stop()

# Model loaded successfully
st.success("✓ Model loaded successfully")

# Input section
st.header("Your Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=18, max_value=120, value=45)
    sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
    bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    exercise = st.slider("Exercise (min/week)", 0, 500, 150, step=10)
    calories = st.number_input(
        "Average calories/day", min_value=0, max_value=10000, value=2000, step=100
    )

with col2:
    fiber = st.number_input(
        "Fiber (g/day)", min_value=0.0, max_value=200.0, value=15.0, step=0.5
    )
    added_sugar = st.number_input(
        "Added sugar (g/day)", min_value=0.0, max_value=500.0, value=30.0, step=1.0
    )
    caffeine = st.number_input(
        "Caffeine (mg/day)", min_value=0, max_value=2000, value=200, step=10
    )
    alcohol = st.number_input(
        "Alcohol (drinks/week)", min_value=0.0, max_value=100.0, value=3.0, step=0.5
    )
    smoker = st.radio("Current smoker", ["No", "Yes"], horizontal=True)

st.divider()
col1, col2 = st.columns(2)

with col1:
    depression = st.number_input(
        "Depression score (PHQ-9, 0-27)",
        min_value=0,
        max_value=27,
        value=3,
        help="Use PHQ-9 score if available, otherwise 0",
    )

with col2:
    systolic = st.number_input(
        "Systolic BP (mmHg)", min_value=70, max_value=250, value=120, step=1
    )
    diastolic = st.number_input(
        "Diastolic BP (mmHg)", min_value=40, max_value=150, value=80, step=1
    )

# Prediction
st.divider()
if st.button("Calculate Risk", type="primary", use_container_width=True):
    try:
        # Build feature dict in same order as feature_names
        input_dict = {
            "age": age,
            "sex": 1 if sex == "Male" else 2,  # NHANES convention: 1=Male, 2=Female
            "BMI": bmi,
            "exercise_min_week": exercise,
            "calories_day": calories,
            "fiber_g_day": fiber,
            "added_sugar_g_day": added_sugar,
            "caffeine_mg_day": caffeine,
            "alcohol_drinks_week": alcohol,
            "current_smoker": 1 if smoker == "Yes" else 0,
            "depression_score": depression,
            "systolic_bp": systolic,
            "diastolic_bp": diastolic,
        }

        # Check that all required features are present
        missing_features = [f for f in feature_names if f not in input_dict]
        if missing_features:
            st.error(f"Missing features in input: {missing_features}")
            st.stop()

        # Order inputs by feature_names (important for model correctness)
        X_ordered = np.array([[input_dict[f] for f in feature_names]])

        # Scale
        X_scaled = scaler.transform(X_ordered)

        # Predict
        risk_prob = model.predict_proba(X_scaled)[0, 1]
        prediction = model.predict(X_scaled)[0]

        st.divider()
        st.subheader("Results")

        # Display risk score
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Sleep Disorder Risk Score",
                f"{risk_prob*100:.1f}%",
                delta=None,
                help="Probability of high sleep disorder risk based on NHANES predictive model",
            )

        with col2:
            risk_category = "HIGH RISK" if risk_prob >= 0.5 else "LOW RISK"
            color = "red" if risk_prob >= 0.5 else "green"
            st.metric(
                "Risk Category",
                risk_category,
                help="Based on 50% threshold",
            )

        st.divider()
        st.subheader("Interpretation")

        if risk_prob >= 0.5:
            st.warning(
                f"⚠️ **High Risk**: Your profile suggests elevated sleep disorder risk ({risk_prob*100:.1f}%). "
                "Consider consulting a sleep specialist for evaluation and polysomnography testing."
            )
        else:
            st.info(
                f"✓ **Low Risk**: Your profile suggests lower sleep disorder risk ({risk_prob*100:.1f}%). "
                "Maintain healthy sleep habits (7-9 hours/night, regular exercise, stress management)."
            )

        # Feature importance note
        st.divider()
        st.subheader("About This Model")
        st.markdown(
            """
            **Model Type:** XGBoost classifier trained on NHANES data (2005-2016)
            
            **Features Used:**
            - Age, Sex (non-modifiable)
            - BMI, Exercise, Diet (calories, fiber, sugar, caffeine), Alcohol, Smoking, Depression, Blood Pressure
            
            **Limitations:**
            - This is a **risk estimate**, NOT a diagnosis
            - Based on U.S. population data; may not generalize to other populations
            - Self-reported measures may have recall bias
            - Should be used alongside clinical evaluation
            
            **Data Source:** CDC NHANES public-use datasets
            
            **For sleep concerns:** Consult a physician or sleep medicine specialist for proper evaluation.
            """
        )

    except Exception as e:
        st.error(f"Error making prediction: {e}")
        import traceback

        st.write(traceback.format_exc())
