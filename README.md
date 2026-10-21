# 🌦️ Weather Prediction App

> Multi-class Weather Classification using Random Forest - 77.8% Accuracy

Live Demo: https://weatherprediction-tpbwt8jnkb9txk5ftud6ag.streamlit.app/

### 📌 Overview
this project predicts - `Sunny, Rainy, Foggy, Drizzle` etc. 
4 Features use: `precipitation, temp_max, temp_min, wind`
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
