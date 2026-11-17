import streamlit as st
import pandas as pd
import pickle

# Load the saved model, scaler, and column names
model = pickle.load(open('loan_model.pkl', 'rb'))
scaler = pickle.load(open('loan_scaler.pkl', 'rb'))
model_columns = pickle.load(open('loan_columns.pkl', 'rb'))

# App Title and Description
st.title("🏦 Loan Approval Prediction App")
st.write("Enter the applicant's details below to check if the loan will be approved.")

# User Input Fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
applicant_income = st.number_input("Applicant Monthly Income", min_value=0, value=5000)
coapplicant_income = st.number_input("Coapplicant Monthly Income", min_value=0, value=0)
loan_amount = st.number_input("Loan Amount (in Thousands, e.g., 120 for 1.2 Lakh)", min_value=1, value=120)
loan_term = st.selectbox("Loan Amount Term (in Months)", [360, 180, 120, 84, 60, 36, 12])
credit_history = st.selectbox("Credit History (1 = Good, 0 = Bad)", [1, 0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Predict Button
if st.button("Check Loan Status"):
    input_data = {
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Married_Yes': 1 if married == "Yes" else 0,
        'Dependents_1': 1 if dependents == "1" else 0,
        'Dependents_2': 1 if dependents == "2" else 0,
        'Dependents_3': 1 if dependents == "3" else 0,
        'Education_Not Graduate': 1 if education == "Not Graduate" else 0,
        'Self_Employed_Yes': 1 if self_employed == "Yes" else 0,
        'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
        'Property_Area_Urban': 1 if property_area == "Urban" else 0
    }
    
    # Convert to DataFrame and match exact column order
    input_df = pd.DataFrame([input_data])[model_columns]
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    
    if prediction == 1:
        st.success("🎉 Congratulations! The Loan is likely to be APPROVED.")
    else:
        st.error("❌ Sorry, the Loan is likely to be REJECTED.")
