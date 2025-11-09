import os
import streamlit as st
import pandas as pd
import warnings
from dotenv import load_dotenv
from src.pipeline.prediction_pipeline import PredictionPipeline

warnings.filterwarnings('ignore')

# Ensure environment variables (e.g., Backblaze credentials) are available when running via Streamlit
load_dotenv()

st.set_page_config(page_title="Customer Categorizer", page_icon="👥", layout="wide")

st.title("🎯 Customer Segmentation System")
st.header("Customer Category Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    education = st.selectbox("Education", ["Basic", "2n Cycle", "Graduation", "Master", "PhD"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Together", "Divorced", "Widow", "Absurd", "YOLO", "Alone"])
    parental_status = st.selectbox("Parental Status", ["Parent", "Non-Parent"])
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
    income = st.number_input("Income", min_value=0, value=50000)
    total_spending = st.number_input("Total Spending", min_value=0, value=500)

with col2:
    days_as_customer = st.number_input("Days as Customer", min_value=0, value=365)
    recency = st.number_input("Recency (days)", min_value=0, value=30)
    wines = st.number_input("Wines Purchases", min_value=0, value=0)
    fruits = st.number_input("Fruits Purchases", min_value=0, value=0)
    meat = st.number_input("Meat Purchases", min_value=0, value=0)
    fish = st.number_input("Fish Purchases", min_value=0, value=0)
    sweets = st.number_input("Sweets Purchases", min_value=0, value=0)

with col3:
    gold = st.number_input("Gold Purchases", min_value=0, value=0)
    web = st.number_input("Web Purchases", min_value=0, value=0)
    catalog = st.number_input("Catalog Purchases", min_value=0, value=0)
    store = st.number_input("Store Purchases", min_value=0, value=0)
    discount_purchases = st.number_input("Discount Purchases", min_value=0, value=0)
    total_promo = st.number_input("Total Promo Accepted", min_value=0, value=0)
    web_visits = st.number_input("Web Visits/Month", min_value=0, value=5)

if st.button("🎯 Predict Category", type="primary", use_container_width=True):
    input_data = [
        age, education, marital_status, parental_status, children, income,
        total_spending, days_as_customer, recency, wines, fruits, meat,
        fish, sweets, gold, web, catalog, store, discount_purchases,
        total_promo, web_visits
    ]
    
    with st.spinner("Analyzing customer profile..."):
        try:
            pipeline = PredictionPipeline()
            prediction = pipeline.run_pipeline(input_data)
            
            st.success("✅ Prediction Complete!")
            st.markdown(f"### Customer Category: **{prediction[0]}**")
            
            category_info = {
                0: {"name": "Low Value", "color": "🔵", "desc": "Budget-conscious customers"},
                1: {"name": "Medium Value", "color": "🟢", "desc": "Regular customers with moderate spending"},
                2: {"name": "High Value", "color": "🟡", "desc": "Premium customers with high engagement"}
            }
            
            if prediction[0] in category_info:
                info = category_info[prediction[0]]
                st.info(f"{info['color']} **{info['name']}**: {info['desc']}")
                
        except Exception as e:
            st.error("❌ Prediction failed. Please check your environment configuration and try again.")
            st.exception(e)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 About")
st.sidebar.info("Customer segmentation system using ML to categorize customers based on behavior and demographics.")
