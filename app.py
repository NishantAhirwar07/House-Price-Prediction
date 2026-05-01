import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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
    return df, model

# Load data and model
try:
    df, model = get_model()
except FileNotFoundError:
    st.error("Error: 'homeprices (1).csv' not found. Please ensure it is in the project directory.")
    st.stop()

# Header
st.title("🏠 House Price Predictor")
st.markdown("Use this tool to estimate the price of a house based on its area.")

# Sidebar
st.sidebar.header("Model Information")
st.sidebar.write(f"Coefficients: {model.coef_[0]:.4f}")
st.sidebar.write(f"Intercept: {model.intercept_:.2f}")

# Input Section
col1, col2 = st.columns([2, 1])
with col1:
    area = st.number_input("Enter House Area (sq ft):", min_value=500, max_value=10000, value=2000, step=100)

with col2:
    st.write("") # Spacer
    st.write("") # Spacer
    predict_btn = st.button("Predict Price")

# Prediction Logic
if predict_btn:
    prediction = model.predict([[area]])
    st.success(f"### The estimated price is: ${prediction[0]:,.2f}")
    st.balloons()

# Visualization
st.divider()
st.subheader("📊 Regression Analysis")
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(df['area'], df['price'], color='#007BFF', label='Actual Data')
ax.plot(df['area'], model.predict(df[['area']]), color='red', label='Prediction Line')
ax.set_xlabel("Area (sq ft)")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)
