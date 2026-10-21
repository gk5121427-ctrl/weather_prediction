# 🌦️ Weather Prediction App

> Multi-class Weather Classification using Random Forest - 77.8% Accuracy

Live Demo: **Coming Soon** (Streamlit pe deploy ke baad yaha link daal dena)

### 📌 Overview
Is project me mausam predict kiya hai - `Sunny, Rainy, Foggy, Drizzle` etc. 
4 Features ka use kiya gaya hai: `precipitation, temp_max, temp_min, wind`
Yeh ek **Classification Problem** hai.

### 📊 Dataset
- `weather` - Target column (categorical)
- `precipitation, temp_max, temp_min, wind` - Features
- `date` - Dropped for model training

### 🤖 Model Details
- **Algorithm:** Random Forest Classifier (`random_state=42`)
- **Preprocessing:** StandardScaler + LabelEncoder
- **Train-Test Split:** 80-20
- **Accuracy:** `77.81%`

```python
rf_model = model.fit(x_train, y_train)
y_pred = rf_model.predict(x_test)
accuracy_score(y_test, y_pred) # 0.7781
