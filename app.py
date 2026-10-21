import streamlit as st
import pickle
import numpy as np

# Cache model - fast load
@st.cache_resource
def load_model():
    model = pickle.load(open('model.pkl','rb'))
    le = pickle.load(open('le.pkl','rb'))
    scaler = pickle.load(open('scaler.pkl','rb'))
    return model, le, scaler

model, le, scaler = load_model()

st.set_page_config(page_title="Weather App", page_icon="🌦️")
st.title("🌦️ Weather Prediction App")
st.divider()

col1, col2 = st.columns(2)

with col1:
    precip = st.slider("Precipitation", 0.0, 20.0, 0.42)
    temp_max = st.slider("Temp Max (°C)", -10.0, 40.0, 12.0)

with col2:
    temp_min = st.slider("Temp Min (°C)", -20.0, 30.0, 5.0)
    wind = st.slider("Wind (km/h)", 0.0, 20.0, 4.0)

st.divider()

if st.button("🔮 Predict Weather", use_container_width=True, type="primary"):
    data = np.array([[precip, temp_max, temp_min, wind]])
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)
    weather = le.inverse_transform(pred)
    
    if weather[0].lower() == 'rain':
        st.error(f"🌧️ Predicted: {weather[0]}")
    elif weather[0].lower() == 'sun':
        st.success(f"☀️ Predicted: {weather[0]}")
    else:
        st.info(f"☁️ Predicted: {weather[0]}")
