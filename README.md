# Smart Lender – AI-Powered Loan Approval Prediction

Smart Lender is a Machine Learning-powered web application that predicts whether a loan applicant is eligible for approval based on financial and personal information. The application assists banks and financial institutions in making faster, consistent, and data-driven lending decisions.

The system trains multiple machine learning models and selects the best-performing model (**XGBoost**) for deployment through a Flask web application.

---

## Features

- Predicts loan approval in real time
- User-friendly web interface built with Flask
- Compares multiple Machine Learning algorithms
- Uses trained model (`model.pkl`) for prediction
- Data preprocessing using a saved scaler (`scaler.pkl`)
- Easy deployment and lightweight architecture

---

## Machine Learning Models

The following algorithms were trained and evaluated:

- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- XGBoost

### Best Model

| Model | Training Accuracy | Testing Accuracy |
|--------|------------------:|-----------------:|
| XGBoost | **94.7%** | **81.1%** |

---

## Technologies Used

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- XGBoost
- NumPy
- Pandas
- SciPy

### Visualization
- Matplotlib
- Seaborn

### Web Framework
- Flask

---

## Project Structure

```
Smart-Lender/
│
├── dataset/
│   └── loan.csv
│
├── model/
│   ├── model.pkl
│   └── scaler.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   ├── home.html
│   ├── predict.html
│   └── submit.html
│
├── app.py
├── train_model.py
├── test_model.py
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Smart-Lender.git
cd Smart-Lender
```

---

### 2. Create a Virtual Environment (Optional)

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run the Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000/
```

---

## Input Features

The model predicts loan approval using:

- Gender
- Marital Status
- Education
- Self Employment
- Applicant Income
- Co-applicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area

---

## Workflow

1. User enters loan applicant details.
2. Flask receives the input.
3. Data is preprocessed using `scaler.pkl`.
4. The trained model (`model.pkl`) predicts loan approval.
5. Prediction result is displayed to the user.

---

## Example Use Cases

### Fast Approval

Applicants with stable income and good credit history are predicted as low-risk, enabling quicker approval.

### High-Risk Detection

Applicants with poor credit history or unstable income are flagged for additional verification.

### Decision Support

The application assists loan officers by providing consistent, data-driven recommendations during the evaluation process.

---

## Requirements

### Hardware

- Intel Core i3 or above
- 4 GB RAM (8 GB Recommended)
- 10 GB Free Storage

### Software

- Windows/Linux/macOS
- Python 3.8+
- VS Code or PyCharm
- Google Chrome or Microsoft Edge

---

## Team Members

| Name | Role |
|------|------|
| Ilipilla Karthik | Team Lead |
| Amani Nallamothu | Team Member |
| Golla Anvitha | Team Member |
| D. Lakshmi Prasanna | Team Member |
| Mamatha Kopuru | Team Member |

---

## Future Enhancements

- Explainable AI using SHAP/LIME
- User Authentication
- REST API Support
- Docker Deployment
- Cloud Deployment (IBM Cloud/AWS)
- Admin Dashboard
- Loan Analytics and Reporting

---

## License

This project is developed for educational and academic purposes.

---

## Acknowledgements

- Scikit-learn
- XGBoost
- Flask
- IBM Cloud
- Open Source Python Community
