import streamlit as st
import pickle
import pandas as pd

# ---- LOAD MODEL ----
data = pickle.load(open('churn_model.pkl', 'rb'))
model = data['model']
scaler = data['scaler']

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📊 Customer Churn Prediction")

# ---- INPUTS ----
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.slider("Tenure (months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])

# IMPORTANT: these are binary in your model (not 3-class)
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])

PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.selectbox("Payment Method", [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check"
])

# ---- CREATE FEATURE DICT (ALL FEATURES) ----
features = {col: 0 for col in model.feature_names_in_}

# ---- FILL NUMERIC + BINARY ----
features['gender'] = 1 if gender == "Male" else 0
features['SeniorCitizen'] = SeniorCitizen
features['Partner'] = 1 if Partner == "Yes" else 0
features['Dependents'] = 1 if Dependents == "Yes" else 0
features['tenure'] = tenure
features['PhoneService'] = 1 if PhoneService == "Yes" else 0

features['MultipleLines'] = 1 if MultipleLines == "Yes" else 0
features['OnlineSecurity'] = 1 if OnlineSecurity == "Yes" else 0
features['OnlineBackup'] = 1 if OnlineBackup == "Yes" else 0
features['DeviceProtection'] = 1 if DeviceProtection == "Yes" else 0
features['TechSupport'] = 1 if TechSupport == "Yes" else 0
features['StreamingTV'] = 1 if StreamingTV == "Yes" else 0
features['StreamingMovies'] = 1 if StreamingMovies == "Yes" else 0

features['PaperlessBilling'] = 1 if PaperlessBilling == "Yes" else 0

features['MonthlyCharges'] = MonthlyCharges
features['TotalCharges'] = TotalCharges

# ---- ONE HOT ENCODING ----
features[f'InternetService_{InternetService}'] = 1
features[f'Contract_{Contract}'] = 1
features[f'PaymentMethod_{PaymentMethod}'] = 1

# ---- DATAFRAME ----
input_df = pd.DataFrame([features])

# ---- SCALE ONLY REQUIRED COLUMNS ----
scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_df[scale_cols] = scaler.transform(input_df[scale_cols])

# ---- PREDICT ----
if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    # Show probability (VERY IMPORTANT for debugging)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_df)[0][1]
        st.write(f"Churn Probability: {prob:.2f}")

    if prediction == 1:
        st.error("⚠️ Customer will churn")
    else:
        st.success("✅ Customer will not churn")