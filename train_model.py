import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# =========================
# Load dataset
# =========================
df = pd.read_csv("data/diabetic_data.csv")

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)


# =========================
# Replace invalid missing values
# =========================
df = df.replace("?", pd.NA)


# =========================
# Create target column
# =========================
# <30 means patient was readmitted within 30 days
# >30 and NO are treated as lower readmission risk
df["target"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)


# =========================
# Select beginner-friendly features
# =========================
selected_features = [
    "age",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "insulin",
    "change",
    "diabetesMed",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id"
]

X = df[selected_features]
y = df["target"]


# =========================
# Separate numeric and categorical columns
# =========================
numeric_features = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id"
]

categorical_features = [
    "age",
    "max_glu_serum",
    "A1Cresult",
    "insulin",
    "change",
    "diabetesMed"
]


# =========================
# Preprocessing
# =========================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# Model
# 8GB RAM friendly settings
# =========================
model = RandomForestClassifier(
    n_estimators=120,
    max_depth=10,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# =========================
# Full pipeline
# =========================
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])


# =========================
# Train model
# =========================
print("Training model...")
pipeline.fit(X_train, y_train)
print("Training completed")


# =========================
# Evaluate model
# =========================
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nModel Evaluation")
print("----------------")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# =========================
# Save files
# =========================
joblib.dump(pipeline, "diabetes_readmission_model.pkl")
joblib.dump(selected_features, "selected_features.pkl")

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "classification_report": classification_report(y_test, y_pred, zero_division=0)
}

joblib.dump(metrics, "model_metrics.pkl")

print("\nModel saved successfully")
print("Selected features saved successfully")
print("Metrics saved successfully")