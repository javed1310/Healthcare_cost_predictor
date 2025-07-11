import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Healthcare Cost Predictor")
st.title("🏥 Healthcare Cost Predictor")
st.write("Estimate your expected medical charges based on lifestyle and demographics.")

if not os.path.exists('gradient_boost_model.pkl') or not os.path.exists('scaler.pkl'):
    st.error("❌ Model or Scaler file not found.")
else:
    model = joblib.load('gradient_boost_model.pkl')
    scaler = joblib.load('scaler.pkl')

    age = st.slider("Age", 18, 65, 30)
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    children = st.slider("Number of Children", 0, 5, 1)
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Do you smoke?", ["yes", "no"])
    region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

    sex_male = 1 if sex == "male" else 0
    smoker_yes = 1 if smoker == "yes" else 0
    region_ne = 1 if region == "northeast" else 0
    region_nw = 1 if region == "northwest" else 0
    region_se = 1 if region == "southeast" else 0

    features = np.array([[age, bmi, children, sex_male, smoker_yes,
                          region_ne, region_nw, region_se]])
    features_scaled = scaler.transform(features)

    if st.button("Predict Medical Cost"):
        prediction = model.predict(features_scaled)[0]
        st.success(f"💰 Estimated Medical Cost: ₹{prediction:,.2f}")
