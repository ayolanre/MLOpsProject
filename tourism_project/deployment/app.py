### Streamlit Deployment App (`app.py`)
'''
This script loads the trained model, builds a Streamlit user interface,
collects customer details, prepares them into a model‑ready format,
and generates a prediction on whether the customer will purchase a travel package.
'''

import os
import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load Model
# ----------------------------
# Build the full path to the saved model inside the deployment folder
model_path = os.path.join(os.path.dirname(__file__), "best_model_v1.pkl")

# Ensure the model exists before loading
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

# Load the trained XGBoost model
model = joblib.load(model_path)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Tourism Package Prediction")
st.write("Fill in customer details to predict whether they will purchase a travel package.")

# Collect user input through interactive widgets
Age = st.slider("Age", 18, 70, 30)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.slider("Duration of Pitch (mins)", 0, 100, 15)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female", "Others"])
NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 5, 2)
NumberOfFollowups = st.slider("Number of Follow-ups", 1, 10, 3)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
NumberOfTrips = st.slider("Number of Trips", 1, 20, 3)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.slider("Number of Children Visiting", 0, 5, 1)
Designation = st.selectbox("Designation", ["Executive", "Manager", "AVP", "VP", "Sr. Manager"])
MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, value=30000.0)

# ----------------------------
# Prepare Input Data
# ----------------------------
# Convert user selections into a single-row DataFrame matching training features
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
    'Passport': 1 if Passport == "Yes" else 0,   # Convert Yes/No to numeric
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,       # Convert Yes/No to numeric
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

# ----------------------------
# Prediction
# ----------------------------
classification_threshold = 0.45  # Same threshold used during training

if st.button("Predict"):
    # Get probability of purchase from the model
    prob = model.predict_proba(input_data)[0, 1]

    # Convert probability into binary prediction
    pred = int(prob >= classification_threshold)

    # Human-readable output
    result = "will purchase the travel package" if pred == 1 else "is unlikely to purchase"

    # Display results
    st.write(f"Prediction: Customer {result}")
    st.write(f"Probability of purchase: {prob:.2f}")
