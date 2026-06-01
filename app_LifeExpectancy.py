import base64
import streamlit as st
import joblib
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns


# Load the trained model and scaler
model = joblib.load("best_random_forest_model.pkl")
scaler = joblib.load("feature_scaler.pkl")
# Streamlit app
# Set page layout

st.set_page_config(layout="wide")

st.image("HeaderImage.png", width=1200)  

st.write("Enter the following features to predict life expectancy:")
# Input fields for features
col1, col2 = st.columns(2)  # Create two columns for better layout
with col1: 
    features = {}
    features['status_developing'] = st.selectbox("Development Status", options=["Developing", "Developed"], index=0)
    features['Adult Mortality'] = st.number_input("Adult Mortality", min_value=0,value=263)
    features['Infant Deaths'] = st.number_input("Infant Deaths", min_value=0,value=62)
    features['Alcohol'] = st.number_input("Alcohol Consumption", min_value=0.0,value=0.01)
    features['Percentage Expenditure'] = st.number_input("Percentage Expenditure", min_value=0.0,value=71.28)
    features['Hepatitis B'] = st.number_input("Hepatitis B Immunization", min_value=0,value=65)
    features['Measles'] = st.number_input("Measles Cases", min_value=0,value=1154)
    features['BMI'] = st.number_input("Body Mass Index (BMI)", min_value=0.0,value=19.1)
    features['Under-Five Deaths'] = st.number_input("Under-Five Deaths", min_value=0,value=83)
    features['Polio'] = st.number_input("Polio Immunization", min_value=0,value=6)
     
with col2:  
    
    features['Total Expenditure'] = st.number_input("Total Expenditure on Health", min_value=0.0,value=8.16)
    features['Diphtheria'] = st.number_input("Diphtheria Immunization", min_value=0,value=65)
    features['HIV/AIDS'] = st.number_input("HIV/AIDS Deaths", min_value=0.0, value=0.1)
    features['GDP'] = st.number_input("Gross Domestic Product (GDP)", min_value=0.0,value=584.259)
    features['Population'] = st.number_input("Population", min_value=0,value=33736494)
    features['Thinness 1-19 years'] = st.number_input("Thinness (1-19 years)", min_value=0.0,value=17.2)
    features['Thinness 5-9 years'] = st.number_input("Thinness (5-9 years)", min_value=0.0,value=17.3)
    features['Income Composition of Resources'] = st.number_input("Income Composition of Resources", min_value=0.0,value=0.479)
    features['Schooling'] = st.number_input("Schooling (years)", min_value=0.0,value=10.1)
    
# Initialize session state
if "show_importances" not in st.session_state:
    st.session_state.show_importances = False
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# Buttons
colA, colB = st.columns(2)
with colA:
    # Predict button    
    if st.button("Predict Life Expectancy"):
        # Prepare the input data for prediction
        input_data = [[
            1 if features['status_developing'] == "Developing" else 0,
            features['Adult Mortality'],
            features['Infant Deaths'],
            features['Alcohol'],
            features['Percentage Expenditure'],
            features['Hepatitis B'],
            features['Measles'],
            features['BMI'],
            features['Under-Five Deaths'],
            features['Polio'],
            features['Total Expenditure'],
            features['Diphtheria'],
            features['HIV/AIDS'],
            features['GDP'],
            features['Population'],
            features['Thinness 1-19 years'],
            features['Thinness 5-9 years'],
            features['Income Composition of Resources'],
            features['Schooling']
        ]]

        scaled_input = scaler.transform(input_data)

        # Predict
        prediction = model.predict(scaled_input)

        st.write(f"Predicted Life Expectancy between: {prediction[0]-1.62:.2f} and {prediction[0]+1.62:.2f} years(considering possible error in prediction).")
        st.session_state.show_importances = False
with colB:
    if st.button("Show Feature Importances"):
        st.session_state.show_importances = True
        feature_importances = model.feature_importances_

# Display results
if st.session_state.prediction is not None:
    st.write(f"Predicted Life Expectancy between: {prediction[0]-1.62:.2f} and {prediction[0]+1.62:.2f} years(considering possible error in prediction).")

if st.session_state.show_importances:
    feature_importances = model.feature_importances_
    
    feature_names = [
            "Development Status",
            "Adult Mortality",
            "Infant Deaths",
            "Alcohol",
            "Percentage Expenditure",
            "Hepatitis B",
            "Measles",
            "BMI",
            "Under-Five Deaths",
            "Polio",
            "Total Expenditure",
            "Diphtheria",
            "HIV/AIDS",
            "GDP",
            "Population",
            "Thinness 1-19 years",
            "Thinness 5-9 years",
            "Income Composition of Resources",
            "Schooling"
        ]
    for i in range(len(feature_importances)):
        st.write(f"{feature_names[i]}: {feature_importances[i]:.4f}")
        
    sns.barplot(x=feature_importances, y=feature_names)
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    st.pyplot(plt)
        