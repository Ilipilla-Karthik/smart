import pandas as pd
import joblib

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

data = pd.DataFrame([{
    "Gender": 1,
    "Married": 0,
    "Dependents": 3,
    "Education": 1,
    "Self_Employed": 1,
    "ApplicantIncome": 1200,
    "CoapplicantIncome": 0,
    "LoanAmount": 350,
    "Loan_Amount_Term": 360,
    "Credit_History": 0,
    "Property_Area": 0
}])

scaled = scaler.transform(data)

prediction = model.predict(scaled)

print("Prediction:", prediction)