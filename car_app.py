import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
model = pickle.load(open('model.pkl', 'rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))

# Title
st.title("🚗 Used Car Price Prediction App")
st.markdown("Enter the details below to predict the **car price** in Pakistan (PKR).")

# User input
title = st.selectbox("Car Title", encoders['title'].classes_)
model_year = st.slider("Model Year", 1990, 2025, 2021)
assembly = st.selectbox("Assembly Type", encoders['assembly'].classes_)
mileage_category = st.selectbox("Mileage Category", encoders['mileage_category'].classes_)
fuel_type = st.selectbox("Fuel Type", encoders['fuel_type'].classes_)
engine_capacity = st.slider("Engine Capacity (cc)", 600, 5000, 1500)
transmission = st.selectbox("Transmission", encoders['transmission'].classes_)
city = st.selectbox("City", encoders['city'].classes_)
color = st.selectbox("Color", encoders['color'].classes_)
vehicle_age = st.slider("Vehicle Age (Years)", 0, 30, 3)
registered = st.slider("Registration Number Code (Numeric)", 1, 100, 38)

# Encode inputs
input_df = pd.DataFrame([{
    'title': encoders['title'].transform([title])[0],
    'model': model_year,
    'assembly': encoders['assembly'].transform([assembly])[0],
    'mileage_category': encoders['mileage_category'].transform([mileage_category])[0],
    'fuel_type': encoders['fuel_type'].transform([fuel_type])[0],
    'engine_capacity': engine_capacity,
    'transmission': encoders['transmission'].transform([transmission])[0],
    'city': encoders['city'].transform([city])[0],
    'color': encoders['color'].transform([color])[0],
    'vehicle_age': vehicle_age,
    'registered': registered
}])

# Predict button
if st.button("Predict Price 💰"):
    predicted_price = model.predict(input_df)[0]
    st.success(f"Estimated Price: **Rs. {int(predicted_price):,}**")
