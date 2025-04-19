# 🚗 Used Car Price Prediction with Machine Learning

A machine learning project that predicts the price of used cars based on their features using a Random Forest Regressor. The dataset consists of 48,000+ records of used cars with various attributes like brand, city, mileage, fuel type, transmission, color, and more.

---

## 📌 Project Objective

To build a machine learning model that can accurately predict the selling price of a used car using relevant features. This helps car dealers, buyers, and sellers make data-driven pricing decisions.

---

## 🧠 ML Model Used
- **Random Forest Regressor** – chosen for its performance, ability to handle nonlinear data, and robustness.

---

## 📂 Dataset Overview
- **Rows:** 48,000+
- **Columns:** Multiple features such as:
  - `title`
  - `assembly`
  - `mileage_category`
  - `fuel_type`
  - `engine_capacity`
  - `transmission`
  - `city`
  - `color`
  - `registered`
  - ...and more.

---

## 🔍 Exploratory Data Analysis (EDA)
Conducted in-depth EDA with visualizations:
- ✅ Heatmap for feature correlation
- ✅ Top most expensive cars
- ✅ City-wise price distribution
- ✅ Fuel-type & transmission-wise trends
- ✅ Outlier detection & feature distributions

---

## 🧪 Model Evaluation
- 📈 Accuracy Score (R²): **93%**
- 📉 RMSE, MAE also calculated
- 📊 **Actual vs Predicted Plot** to visualize model performance

> 🔴 A red dashed line shows perfect prediction line. The closer the blue dots are to this line, the better the model.

---

## 🛠 Features Engineering
- Label Encoding for categorical features using `LabelEncoder`
- Data cleaned & preprocessed
- Train-test split performed

---

## 🧾 Sample Prediction
```python
example_df = pd.DataFrame([{
    'title': encoders['title'].transform(['Honda City Aspire Prosmatec 1.5 i-VTEC 2016'])[0],
    'assembly': encoders['assembly'].transform(['Local'])[0],
    'mileage_category': encoders['mileage_category'].transform(['Low'])[0],
    'fuel_type': encoders['fuel_type'].transform(['Petrol'])[0],
    'engine_capacity': 1300,
    'transmission': encoders['transmission'].transform(['Automatic'])[0],
    'city': encoders['city'].transform(['Karachi'])[0],
    'color': encoders['color'].transform(['White'])[0],
    'registered': encoders['registered'].transform(['Islamabad'])[0]
}])

predicted_price = model.predict(example_df)
print("Predicted Price: Rs.", int(predicted_price[0]))
