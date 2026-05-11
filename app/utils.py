import joblib

model = joblib.load("app/model.pkl")
scaler = joblib.load("app/scaler.pkl")


def predict_transaction(data):
    scaled = scaler.transform([data])
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": float(probability)
    }