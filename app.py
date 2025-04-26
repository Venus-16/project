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

# Your original style inputs
income = st.number_input('Annual Income ($)', min_value=0, value=50000)
credit_score = st.slider('Credit Score', 300, 850, 700)
loan_amount = st.number_input('Loan Amount ($)', min_value=0, value=10000)
dti_ratio = st.slider('Debt-to-Income Ratio (%)', 0.0, 100.0, 30.0)
employment_status = st.selectbox('Employment Status', ['employed', 'unemployed'])

# Predict button
if st.button('Predict Approval'):
    # Prepare input correctly
    employment_encoded = status_encoder.transform([employment_status])[0]
    
    # Important: DataFrame columns must match **exactly the same order** as model trained
    input_data = pd.DataFrame([[
        income,
        credit_score,
        loan_amount,
        dti_ratio,
        employment_encoded
    ]], columns=['Income', 'Credit_Score', 'Loan_Amount', 'DTI_Ratio', 'Employment_Status'])

    # Predict
    prediction = model.predict(input_data)
    prediction_label = approval_encoder.inverse_transform(prediction)[0]

    # Display result
    if prediction_label == 'Yes':
        st.success("🎉 Loan Approved!")
        st.balloons()
    else:
        st.error("❌ Loan Rejected!")



