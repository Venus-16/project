# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 21:50:21 2025

@author: User
"""

import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load saved model
model = pickle.load(open('loan_approval_model.pkl', 'rb'))

# Recreate and load encoders
status_encoder = LabelEncoder()
status_encoder.classes_ = np.array(['employed', 'unemployed'])  # Adjust according to your training

approval_encoder = LabelEncoder()
approval_encoder.classes_ = np.array(['No', 'Yes'])  # Adjust according to your training

# Streamlit UI
st.title("🏦 Loan Approval Prediction App")
st.header("Project2_Group9")

st.subheader("Enter Customer Details:")

income = st.number_input('Annual Income ($)', min_value=20000, max_value=200000, step=1000)
credit_score = st.slider('Credit Score', 300, 850, 600)
loan_amount = st.number_input('Loan Amount ($)', min_value=1000, max_value=160000, step=1000)
dti_ratio = st.slider('Debt-to-Income Ratio (%)', 0.0, 250.0, 30.0)
employment_status = st.selectbox('Employment Status', ['employed', 'unemployed'])

# Predict button
if st.button('Predict Approval'):
    # Prepare the input
    status_encoded = status_encoder.transform([employment_status])[0]
    input_data = np.array([[income, credit_score, loan_amount, dti_ratio, status_encoded]])
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_label = approval_encoder.inverse_transform(prediction)[0]
    
    # Show result nicely
    if prediction_label == 'Yes':
        st.success("🎉 Loan Approved!")
    else:
        st.error("❌ Loan Rejected!")

