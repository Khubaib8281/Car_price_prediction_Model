import streamlit as st
import pandas as pd
import joblib
import gdown
import os
from PIL import Image
# File paths
model_path = 'model_compressed.pkl'
encoders_path = 'encoders.pkl'

# Download model if not already downloaded
file_id = '1uu16nwqElnQey42iLGG2FZfohXCOYjUH'
url = f'https://drive.google.com/uc?id={file_id}'

if not os.path.exists(model_path):
    try:
        st.info("Downloading model file...")
        gdown.download(url, model_path, quiet=False)
        st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        st.stop()

# Load model and encoders
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

try:
    encoders = joblib.load(encoders_path)
except Exception as e:
    st.error(f"Failed to load encoders.pkl. Make sure the file is in the same folder.\nError: {e}")
    st.stop()

# Streamlit UI
st.title("🚗 Used Cars Price Prediction")
st.markdown("""
    ## ⚠️ Disclaimer:
    
    **Please note**: The car price predictions provided by this model are **estimated values** based on available data up to **2024**. Actual prices may vary depending on factors such as market fluctuations, condition of the car, location, and other external influences. This tool is intended for informational purposes and should not be considered as a definitive price guide.
""")
st.markdown("### **Enter the details below to predict the **car price** in Pakistan (PKR)**.")

# Input fields
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
registered = st.selectbox("Registered city/ Un-registered", encoders['registered'].classes_)

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
    'registered': encoders['registered'].transform([registered])[0]
}])

# Predict
if st.button("Predict Price 💰"):
    try:
        predicted_price = model.predict(input_df)[0]
        st.success(f"Estimated Price: **Rs. {int(predicted_price):,}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("### **Some useful Visualizations**")
st.markdown("#### **Feature Importance**")

image = Image.open('feature_imp.png')
st.image(image, caption='Feature Importance', use_container_width=True)

image = Image.open('cities_with_most_cars.png')
st.image(image, caption='Cities with Most Cars', use_container_width=True)

image = Image.open('pie_chart_on_most_colors.png')
st.image(image, caption='Most Common Car Colors', use_container_width=True)

image = Image.open('heatmap.png')
st.image(image, caption='Features Corelation', use_container_width=True)