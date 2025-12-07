import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="UncloudMe/Tourism-Project", filename="best_tourism_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Buyer Prediction System")
st.write("""
This application predicts potential buyers, and enhances decision-making for marketing strategies.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
Age = st.number_input("Customer Age", min_value=18, max_value=100,  step=1)
TypeofContact= st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
CityTier = st.number_input("City Tier", min_value=1, max_value=3)
DurationOfPitch = st.number_input("Duration Of Pitch", min_value=1, max_value=180)
Occupation= st.selectbox("Occupation", ["Salaried", "Free Lancer","Small Business","Large Business"])
Gender= st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting", min_value=1, max_value=5)
NumberOfFollowups = st.number_input("Number Of Followups", min_value=1, max_value=10)
ProductPitched= st.selectbox("Product Pitched", ["Basic", "Deluxe","Standard","King","Super Deluxe"])
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=3, max_value=5)
MaritalStatus= st.selectbox("Marital Status", ["Single", "Marrried","Unmarrried","Divorced"])
NumberOfTrips = st.number_input("Number Of Trips", min_value=0, max_value=50)
Passport=st.number_input("Passport", min_value=0, max_value=1)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5)
OwnCar = st.number_input("Own Car", min_value=0, max_value=1)
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=5, value=0)
Designation= st.selectbox("Designation", ["Manager", "Senior Manager","Executive","VP","AVP"])
MonthlyIncome = st.number_input("MonthlyIncome", min_value=0, max_value=100000)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])


if st.button("Predict Customer Potential"):
    prediction = model.predict(input_data)[0]
    result = "A Potential Customer" if prediction == 1 else "Not a potential customer"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
