from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load Model and Scaler
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")


@app.route("/")
def home():return "<h1>Flask is Working!</h1>"


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/submit", methods=["POST"])
def submit():

    try:
        # -------------------------
        # Read Form Data
        # -------------------------
        gender = 1 if request.form["Gender"] == "Male" else 0
        married = 1 if request.form["Married"] == "Yes" else 0
        dependents = int(request.form["Dependents"])
        education = 0 if request.form["Education"] == "Graduate" else 1
        self_employed = 1 if request.form["Self_Employed"] == "Yes" else 0

        applicant_income = float(request.form["ApplicantIncome"])
        coapplicant_income = float(request.form["CoapplicantIncome"])
        loan_amount = float(request.form["LoanAmount"])
        loan_term = float(request.form["Loan_Amount_Term"])

        credit_history = int(request.form["Credit_History"])

        area = request.form["Property_Area"]

        if area == "Rural":
            property_area = 0
        elif area == "Semiurban":
            property_area = 1
        else:
            property_area = 2

        # -------------------------
        # Create DataFrame
        # -------------------------
        data = pd.DataFrame([{
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area
        }])

        print("\n========== INPUT DATA ==========")
        print(data)

        # -------------------------
        # Scale
        # -------------------------
        scaled_data = scaler.transform(data)

        print("\n========== SCALED DATA ==========")
        print(scaled_data)

        # -------------------------
        # Predict
        # -------------------------
        prediction = model.predict(scaled_data)

        print("\n========== PREDICTION ==========")
        print(prediction)

        if prediction[0] == 1:
            result = "🎉 Congratulations! Loan Approved"
        else:
            result = "❌ Sorry! Loan Rejected"

        return render_template("submit.html", result=result)

    except Exception as e:
        print("\n========== ERROR ==========")
        print(e)
        return f"<h2>Error:</h2><pre>{e}</pre>"


if __name__ == "__main__":
    app.run(debug=True)