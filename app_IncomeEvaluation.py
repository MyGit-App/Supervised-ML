import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------------
# LOAD ARTIFACTS
# ------------------------------------------------------------
model = joblib.load("best_model_GradientBoosting.pkl")
scaler = joblib.load("scaler_income.pkl")
training_columns = joblib.load("training_columns.pkl")

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Income Prediction App",
    page_icon="💰",
    layout="wide"
)

st.markdown("""
    <style>
        .main { background-color: #F7F9FC; }
        .stButton>button {
            background-color:#4CAF50;
            color:white;
            border-radius:8px;
            height:3em;
            width:12em;
            font-size:16px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.title("⚙️ App Controls")
st.sidebar.info("Fill the details on the right to predict income.")

st.sidebar.markdown("### About")
st.sidebar.write("""
This app predicts whether a person earns **>50K** or **<=50K**  
based on demographic and work-related attributes.
""")

# ------------------------------------------------------------
# MAIN TITLE
# ------------------------------------------------------------
st.title("💼 Income Prediction System")
st.write("Provide the details below and get an instant prediction.")

# ------------------------------------------------------------
# INPUT FORM
# ------------------------------------------------------------
st.subheader("📝 User Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 17, 90, 30)
    education_num = st.number_input("Education-num", 1, 16, 10)
    hours_per_week = st.number_input("Hours per week", 1, 99, 40)

with col2:
    workclass = st.selectbox("Workclass", [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked"
    ])

    marital_status = st.selectbox("Marital Status", [
        "Married-civ-spouse", "Divorced", "Never-married", "Separated",
        "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ])
    relationship = st.selectbox("Relationship", [
        "Wife", "Own-child", "Husband", "Not-in-family",
        "Other-relative", "Unmarried"
    ])

with col3:
    occupation = st.selectbox("Occupation", [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv",
        "Armed-Forces"
    ])
    race = st.selectbox("Race", [
        "White", "Black", "Asian-Pac-Islander",
        "Amer-Indian-Eskimo", "Other"
    ])
    sex = st.selectbox("Sex", ["Male", "Female"])
    native_country = st.selectbox("Native Country", [
        "United-States", "Mexico", "Greece", "Vietnam", "India", "China",
        "Japan", "England", "Canada", "Germany", "Philippines", "Italy",
        "Cuba", "Jamaica", "South", "Puerto-Rico", "Honduras"
    ])

capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.number_input("Capital Loss", 0, 5000, 0)

# ------------------------------------------------------------
# PREDICTION BUTTON
# ------------------------------------------------------------
predict_btn = st.button("🔍 Predict Income")

# ------------------------------------------------------------
# PROCESS INPUT & PREDICT
# ------------------------------------------------------------
if predict_btn:

    input_dict = {
        "age": age,
        "workclass": workclass,
        "education-num": education_num,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country
    }

    input_df = pd.DataFrame([input_dict])

    # Clean categorical spaces
    for col in input_df.select_dtypes(include="object").columns:
        input_df[col] = input_df[col].str.strip()

    #st.write(training_columns  )
    # Encode
    input_encoded = pd.get_dummies(input_df)

    #st.write(input_encoded.columns)
    # Align with training columns
    input_aligned = input_encoded.reindex(columns=training_columns, fill_value=0)

    #st.write(input_aligned.columns)

    input_aligned = scaler.transform(input_aligned)

    
    # Predict
    pred = model.predict(input_aligned)[0]

    # ------------------------------------------------------------
    # OUTPUT CARD
    # ------------------------------------------------------------
    st.subheader("📌 Prediction Result")

    if pred == 1:
        st.success("### 💰 Predicted Income: **>50K**")
    else:
        st.info("### 📉 Predicted Income: **<=50K**")