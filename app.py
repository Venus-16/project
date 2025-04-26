# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 21:50:21 2025

@author: User
"""

import streamlit as st
import pickle
import pandas as pd

# Load saved model and encoders
model = pickle.load(open('loan_approval_model.pkl', 'rb'))
status_encoder = pickle.load(open('employment_status_encoder.pkl', 'rb'))
approval_encoder = pickle.load(open('approval_encoder.pkl', 'rb'))

# Streamlit UI
st.title("🏦 Loan Approval Prediction App")
st.header("Project2_Group9")
st.subheader("Enter Customer Details:")

# Create input form
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input('Annual Income ($)', min_value=20000, max_value=100000, value=50000, step=1000)
        loan_amount = st.number_input('Loan Amount ($)', min_value=1000, max_value=50000, value=10000, step=500)

    with col2:
        credit_score = st.slider('Credit Score', 300, 850, 700)
        dti_ratio = st.slider('Debt-to-Income Ratio (%)', 0.0, 100.0, 30.0)

    employment_status = st.selectbox('Employment Status', ['employed', 'unemployed'])

    # Submit button inside form
    submit = st.form_submit_button("Predict Approval")

# Predict when button is clicked
if submit:
    # Prepare the input
    status_encoded = status_encoder.transform([employment_status])[0]

    input_data = pd.DataFrame({
        'Income': [income],
        'Credit_Score': [credit_score],
        'Loan_Amount': [loan_amount],
        'DTI_Ratio': [dti_ratio],
        'Employment_Status': [status_encoded]
    })

    # Make prediction
    prediction = model.predict(input_data)
    prediction_label = approval_encoder.inverse_transform(prediction)[0]

    # Show result nicely
    st.subheader("Prediction Result:")
    if prediction_label == 'Yes':
        st.success("🎉 Loan Approved!")
        st.balloons()
    else:
        st.error("❌ Loan Rejected!")


