import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Page Config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

# Custom CSS for a clean look
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { width: 100%; border-radius: 5px; background-color: #007BFF; color: white; }
    </style>
    """, unsafe_allow_html=True)

# Function to load and train (Cached)
@st.cache_resource
def get_model():
    # Ensure 'homeprices (1).csv' is in your folder
    df = pd.read_csv('homeprices (1).csv')
    X = df[['area']]
    Y = df['price']
    model = LinearRegression()
    model.fit(X, Y)
    return model

# Load model
try:
    model = get_model()
except FileNotFoundError:
    st.error("Error: 'homeprices (1).csv' not found. Please ensure it is in the project directory.")
    st.stop()

# Header
st.title("🏠 House Price Predictor")
st.markdown("Enter the area (sq ft) below to get a price estimation.")

# Input Section
area = st.number_input("Enter House Area (sq ft):", min_value=500, max_value=10000, value=2000, step=100)

predict_btn = st.button("Predict Price")

# Prediction Logic
if predict_btn:
    prediction = model.predict([[area]])
    st.success(f"### The estimated price is: ${prediction[0]:,.2f}")
    st.balloons()
