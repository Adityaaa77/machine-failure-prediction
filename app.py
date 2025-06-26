# app.py – Streamlit App
import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# App title
st.set_page_config(page_title="Predictive Maintenance", layout="centered")
st.title("🔧 Predictive Maintenance – Machine Failure Detection")

# User inputs
st.header("📥 Enter Sensor Readings:")
temp = st.number_input("Temperature", value=25.0)
pressure = st.number_input("Pressure", value=100.0)
vibration = st.number_input("Vibration", value=0.5)
rpm = st.number_input("RPM", value=500.0)

# Predict button
if st.button("🔍 Predict Failure"):
    user_input = np.array([[temp, pressure, vibration, rpm]])
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]

    st.subheader("📊 Prediction:")
    if prediction == 1:
        st.error("⚠️ Machine Failure Likely!")
    else:
        st.success("✅ No Failure Detected.")

# Footer
st.markdown("---")
st.markdown("Built by Aditya Rajpal using Streamlit")
