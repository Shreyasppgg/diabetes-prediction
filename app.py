import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# App title
st.title("🩺 Diabetes Prediction App")
st.write("Enter the details below to predict whether a person has diabetes.")

# Load dataset
df = pd.read_csv("dataset.csv")

# Split features and target
X = df.drop(columns='Outcome', axis=1)
y = df['Outcome']

# Data preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Model accuracy
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

st.sidebar.header("Model Information")
st.sidebar.write(f"Training Accuracy: {train_acc:.2f}")
st.sidebar.write(f"Testing Accuracy: {test_acc:.2f}")

# User input form
st.subheader("Enter Patient Details")
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Prediction
if st.button("Predict"):
    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, dpf, age]])
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)

    if prediction[0] == 1:
        st.error("⚠️ The model predicts that the person **has diabetes**.")
    else:
        st.success("✅ The model predicts that the person **does not have diabetes**.")
