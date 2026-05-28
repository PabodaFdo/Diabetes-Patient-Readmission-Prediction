import streamlit as st
import pandas as pd
import joblib
from datetime import datetime


# =========================
# Load model and files
# =========================
model = joblib.load("diabetes_readmission_model.pkl")
selected_features = joblib.load("selected_features.pkl")
metrics = joblib.load("model_metrics.pkl")


# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Diabetes Readmission Prediction",
    page_icon="🏥",
    layout="wide"
)


# =========================
# Session state
# =========================
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# =========================
# Helper functions
# =========================
def get_risk_level(probability):
    if probability < 40:
        return "Low Risk"
    elif probability < 70:
        return "Medium Risk"
    else:
        return "High Risk"


def get_recommendation(risk_level):
    if risk_level == "Low Risk":
        return """
        The patient has a lower possibility of readmission within 30 days.
        Normal discharge instructions and routine follow-up may be enough.
        """
    elif risk_level == "Medium Risk":
        return """
        The patient may need additional follow-up after discharge.
        A doctor or hospital staff member should review the patient record carefully.
        """
    else:
        return """
        The patient has a higher possibility of readmission within 30 days.
        Close monitoring, early follow-up, and proper discharge planning are recommended.
        """


# =========================
# Sidebar
# =========================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Home", "Predict Readmission", "Analytics", "Model Performance", "Prediction History"]
)


# =========================
# Home Page
# =========================
if page == "Home":
    st.title("🏥 AI-Based Diabetes Patient Readmission Prediction System")

    st.write("""
    This machine learning system predicts whether a diabetes patient is likely to be
    readmitted to the hospital within 30 days after discharge.
    """)

    st.subheader("Project Goal")

    st.write("""
    The main goal of this project is to support hospital decision-making by identifying
    patients who may need extra follow-up care after discharge.
    """)

    st.subheader("Risk Levels")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("Low Risk")
        st.write("Patient is less likely to return within 30 days.")

    with col2:
        st.warning("Medium Risk")
        st.write("Patient may need scheduled follow-up care.")

    with col3:
        st.error("High Risk")
        st.write("Patient may need close monitoring after discharge.")

    st.info("Disclaimer: This project is for educational purposes only and should not be used as real medical advice.")


# =========================
# Prediction Page
# =========================
elif page == "Predict Readmission":
    st.title("🔍 Predict Diabetes Patient Readmission Risk")

    st.write("Enter patient hospital details below:")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.selectbox(
            "Age Group",
            ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
             "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
        )

        time_in_hospital = st.number_input(
            "Time in Hospital",
            min_value=1,
            max_value=14,
            value=3
        )

        num_lab_procedures = st.number_input(
            "Number of Lab Procedures",
            min_value=0,
            max_value=150,
            value=40
        )

        num_procedures = st.number_input(
            "Number of Procedures",
            min_value=0,
            max_value=10,
            value=1
        )

        num_medications = st.number_input(
            "Number of Medications",
            min_value=0,
            max_value=100,
            value=15
        )

    with col2:
        number_outpatient = st.number_input(
            "Number of Outpatient Visits",
            min_value=0,
            max_value=50,
            value=0
        )

        number_emergency = st.number_input(
            "Number of Emergency Visits",
            min_value=0,
            max_value=50,
            value=0
        )

        number_inpatient = st.number_input(
            "Number of Inpatient Visits",
            min_value=0,
            max_value=50,
            value=0
        )

        number_diagnoses = st.number_input(
            "Number of Diagnoses",
            min_value=1,
            max_value=20,
            value=5
        )

        max_glu_serum = st.selectbox(
            "Glucose Serum Test Result",
            ["None", "Norm", ">200", ">300"]
        )

    with col3:
        A1Cresult = st.selectbox(
            "HbA1c Result",
            ["None", "Norm", ">7", ">8"]
        )

        insulin = st.selectbox(
            "Insulin Status",
            ["No", "Steady", "Up", "Down"]
        )

        change = st.selectbox(
            "Medication Change",
            ["No", "Ch"]
        )

        diabetesMed = st.selectbox(
            "Diabetes Medication",
            ["No", "Yes"]
        )

        admission_type_id = st.number_input(
            "Admission Type ID",
            min_value=1,
            max_value=8,
            value=1
        )

        discharge_disposition_id = st.number_input(
            "Discharge Disposition ID",
            min_value=1,
            max_value=30,
            value=1
        )

        admission_source_id = st.number_input(
            "Admission Source ID",
            min_value=1,
            max_value=25,
            value=7
        )

    if st.button("Predict Readmission Risk"):
        input_data = pd.DataFrame({
            "age": [age],
            "time_in_hospital": [time_in_hospital],
            "num_lab_procedures": [num_lab_procedures],
            "num_procedures": [num_procedures],
            "num_medications": [num_medications],
            "number_outpatient": [number_outpatient],
            "number_emergency": [number_emergency],
            "number_inpatient": [number_inpatient],
            "number_diagnoses": [number_diagnoses],
            "max_glu_serum": [max_glu_serum],
            "A1Cresult": [A1Cresult],
            "insulin": [insulin],
            "change": [change],
            "diabetesMed": [diabetesMed],
            "admission_type_id": [admission_type_id],
            "discharge_disposition_id": [discharge_disposition_id],
            "admission_source_id": [admission_source_id]
        })

        probability = model.predict_proba(input_data)[0][1] * 100
        prediction = model.predict(input_data)[0]

        risk_level = get_risk_level(probability)
        recommendation = get_recommendation(risk_level)

        st.subheader("Prediction Result")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Readmission Probability", f"{probability:.2f}%")

        with col_b:
            if prediction == 1:
                st.metric("Prediction", "Likely Readmission")
            else:
                st.metric("Prediction", "Lower Readmission Possibility")

        with col_c:
            st.metric("Risk Level", risk_level)

        if risk_level == "Low Risk":
            st.success(risk_level)
        elif risk_level == "Medium Risk":
            st.warning(risk_level)
        else:
            st.error(risk_level)

        st.subheader("Recommendation")
        st.write(recommendation)

        history_record = {
            "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Age": age,
            "Time in Hospital": time_in_hospital,
            "Lab Procedures": num_lab_procedures,
            "Medications": num_medications,
            "Inpatient Visits": number_inpatient,
            "Emergency Visits": number_emergency,
            "Diagnoses": number_diagnoses,
            "Readmission Probability (%)": round(probability, 2),
            "Risk Level": risk_level
        }

        st.session_state.prediction_history.append(history_record)


# =========================
# Analytics Page
# =========================
elif page == "Analytics":
    st.title("📊 Dataset Analytics")

    try:
        df = pd.read_csv("data/diabetic_data.csv")
        df["target"] = df["readmitted"].apply(lambda x: "Readmitted <30 Days" if x == "<30" else "Not Readmitted <30 Days")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Dataset Shape")
        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

        st.subheader("Readmission Distribution")
        target_counts = df["target"].value_counts()
        st.bar_chart(target_counts)

        st.subheader("Age Group Distribution")
        age_counts = df["age"].value_counts().sort_index()
        st.bar_chart(age_counts)

        st.subheader("Average Values")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Average Time in Hospital", round(df["time_in_hospital"].mean(), 2))

        with col2:
            st.metric("Average Medications", round(df["num_medications"].mean(), 2))

        with col3:
            st.metric("Average Diagnoses", round(df["number_diagnoses"].mean(), 2))

    except FileNotFoundError:
        st.error("Dataset file not found. Please place diabetic_data.csv inside the data folder.")


# =========================
# Model Performance Page
# =========================
elif page == "Model Performance":
    st.title("📈 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2f}")

    with col2:
        st.metric("Precision", f"{metrics['precision']:.2f}")

    with col3:
        st.metric("Recall", f"{metrics['recall']:.2f}")

    with col4:
        st.metric("F1 Score", f"{metrics['f1_score']:.2f}")

    st.subheader("Confusion Matrix")
    st.write(metrics["confusion_matrix"])

    st.subheader("Classification Report")
    st.text(metrics["classification_report"])

    st.info("""
    Recall is important in this project because the system should reduce the chance
    of missing patients who are actually at high risk of readmission.
    """)


# =========================
# Prediction History Page
# =========================
elif page == "Prediction History":
    st.title("🕘 Prediction History")

    if len(st.session_state.prediction_history) == 0:
        st.info("No predictions made yet.")
    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df)

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction History as CSV",
            data=csv,
            file_name="diabetes_readmission_prediction_history.csv",
            mime="text/csv"
        )