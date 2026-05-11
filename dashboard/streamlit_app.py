import streamlit as st
import requests

st.title("🏦 Bank Fraud Detection Dashboard")

st.write("Enter transaction details below")

amount = st.number_input("Transaction Amount", min_value=0.0)
time = st.number_input("Transaction Time", min_value=0.0)

values = []

for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0)
    values.append(val)

if st.button("Predict Fraud"):
    payload = {
        "Time": time,
        "Amount": amount
    }

    for i in range(1, 29):
        payload[f"V{i}"] = values[i - 1]

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=payload
    )

    result = response.json()

    if result["fraud_prediction"] == 1:
        st.error(f"Fraud Detected! Probability: {result['fraud_probability']:.2f}")
    else:
        st.success(f"Legitimate Transaction. Probability: {result['fraud_probability']:.2f}")