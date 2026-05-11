from fastapi import FastAPI
from app.predict import Transaction
from app.utils import predict_transaction

# Create FastAPI app
app = FastAPI(
    title="Bank Fraud Detection API",
    description="Fraud Detection using Machine Learning",
    version="1.0"
)

# Home route
@app.get("/")
def home():
    return {"message": "Fraud Detection API Running Successfully"}

# Prediction route
@app.post("/predict")
def predict(transaction: Transaction):

    data = [
        transaction.Time,
        transaction.V1,
        transaction.V2,
        transaction.V3,
        transaction.V4,
        transaction.V5,
        transaction.V6,
        transaction.V7,
        transaction.V8,
        transaction.V9,
        transaction.V10,
        transaction.V11,
        transaction.V12,
        transaction.V13,
        transaction.V14,
        transaction.V15,
        transaction.V16,
        transaction.V17,
        transaction.V18,
        transaction.V19,
        transaction.V20,
        transaction.V21,
        transaction.V22,
        transaction.V23,
        transaction.V24,
        transaction.V25,
        transaction.V26,
        transaction.V27,
        transaction.V28,
        transaction.Amount
    ]

    result = predict_transaction(data)

    return result