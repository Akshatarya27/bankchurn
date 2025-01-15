import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the pre-trained models
with open('LR_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('standard_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title('Customer Churn Prediction')

# Input fields for new customer data (in two columns)
col1, col2 = st.columns(2)

with col1:
    country = st.selectbox("Select Country", options=["France", "Spain", "Germany"])
    gender = st.selectbox("Select Gender", options=["Male", "Female"])
    age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)
    products_number = st.number_input("Enter Number of Products", min_value=1, max_value=10, value=2)

with col2:
    tenure = st.number_input("Enter Tenure", min_value=0, max_value=100, value=5)
    balance = st.number_input("Enter Balance", min_value=0.0, max_value=1000000.0, value=10000.0)
    credit_card = st.selectbox("Has Credit Card", options=["No", "Yes"])
    active_member = st.selectbox("Is Active Member", options=["No", "Yes"])
    estimated_salary = st.number_input("Enter Estimated Salary", min_value=0.0, max_value=1000000.0, value=50000.0)

# Convert categorical data to numeric
country_map = {"France": 0, "Spain": 2, "Germany": 1}
country_numeric = country_map[country]

gender_map = {"Female": 0, "Male": 1}
gender_numeric = gender_map[gender]

credit_map = {"No": 0, "Yes": 1}
credit_numeric = credit_map[credit_card]

active_map = {"No": 0, "Yes": 1}
active_numeric = active_map[active_member]

# Create DataFrame from user input
input_data = pd.DataFrame({
    'credit_score': [np.random.randint(300, 850)],  # Example random credit score
    'country': [country_numeric],
    'gender': [gender_numeric],
    'age': [age],
    'tenure': [tenure],
    'balance': [balance],
    'products_number': [products_number],
    'credit_card': [credit_numeric],
    'active_member': [active_numeric],
    'estimated_salary': [estimated_salary]
})

# Preprocess input data: Scale using the loaded scaler
input_data_scaled = scaler.transform(input_data)

# Add a submit button
if st.button("Submit"):
    # Make prediction using the model
    churn_prediction = rf_model.predict(input_data_scaled)

    # Display result with enlarged font
    if churn_prediction[0] == 1:
        st.markdown(
            '<h3 style="color: red;">Prediction: Customer is likely to leave the Bank.</h3>', 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<h3 style="color: green;">Prediction: Customer is unlikely to leave the Bank.</h3>', 
            unsafe_allow_html=True
        )
