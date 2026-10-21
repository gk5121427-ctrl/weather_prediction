import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('model.pkl','rb'))
le = pickle.load(open('le.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

st.title("🌦️ Weather Prediction App")
st.write("Random Forest - 77.8% Accuracy")

precip = st.number_input("Precipitation")
temp_max = st.number_input("Temp Max")
temp_min = st.number_input("Temp Min")
wind = st.number_input("Wind")

if st.button("Predict Weather"):
    data = np.array([[precip, temp_max, temp_min, wind]])
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)
    weather = le.inverse_transform(pred)
    st.success(f"Predicted Weather: {weather[0]}")