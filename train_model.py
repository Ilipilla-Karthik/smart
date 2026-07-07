import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ==============================
# Load Dataset
# ==============================
df = pd.read_csv("dataset/loan.csv")

print("\nDataset Columns:")
print(df.columns.tolist())

# ==============================
# Handle Missing Values
# ==============================
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# ==============================
# Encode Categorical Columns
# ==============================
encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

# ==============================
# Features & Target
# ==============================
X = df.drop(["Loan_ID", "Loan_Status"], axis=1)
y = df["Loan_Status"]

print("\nFeatures Used:")
print(X.columns.tolist())

# ==============================
# Balance Dataset
# ==============================
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# ==============================
# Feature Scaling
# ==============================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==============================
# Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# Train Decision Tree
# ==============================
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ==============================
# Prediction
# ==============================
y_pred = model.predict(X_test)
print("\nUnique Predictions:")
print(set(y_pred))

print("\nFirst 20 Predictions:")
print(y_pred[:20])

print("\nFirst 20 Actual:")
print(y_test.values[:20])

# ==============================
# Evaluation
# ==============================
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("Decision Tree Performance")
print("==============================")
print(f"Accuracy : {accuracy:.2%}")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

# ==============================
# Save Model & Scaler
# ==============================
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\n✅ Model Saved Successfully!")
print("✅ Scaler Saved Successfully!")