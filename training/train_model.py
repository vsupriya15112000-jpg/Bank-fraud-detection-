import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Create app folder automatically
os.makedirs("app", exist_ok=True)

# Load dataset
df = pd.read_csv("data/fraud.csv")

print("Dataset Loaded Successfully")

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle imbalance
smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("SMOTE Applied")

# Train model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train_resampled, y_train_resampled)

print("Model Training Completed")

# Predictions
predictions = model.predict(X_test_scaled)

print(classification_report(y_test, predictions))

# Save files
joblib.dump(model, "app/model.pkl")
joblib.dump(scaler, "app/scaler.pkl")

print("model.pkl saved successfully")
print("scaler.pkl saved successfully")