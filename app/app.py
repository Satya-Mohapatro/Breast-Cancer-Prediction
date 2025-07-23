import streamlit as st
import pandas as pd
import joblib

# Load pipeline and feature list
pipeline = joblib.load('../models/logreg_model.pkl')

# Features expected by the pipeline
expected_features = [
 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst',
 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
 'compactness_worst', 'concavity_worst', 'concave points_worst',
 'symmetry_worst', 'fractal_dimension_worst'
]

st.set_page_config(page_title='Breast Cancer Prediction', layout='centered')

st.title("Breast Cancer Prediction App")
st.write("""
Enter **diagnostic measurements** to predict whether a breast tumor is **Malignant or Benign**.
""")

# User-friendly subset for input:
user_input_features = [
    'radius_mean',
    'perimeter_mean',
    'area_mean',
    'concavity_mean',
    'concave points_mean',
    'texture_mean',
    'radius_worst',
    'perimeter_worst',
    'area_worst'
]

input_data = {}

st.subheader("Enter Diagnostic Measurements:")

col1, col2, col3 = st.columns(3)
for idx, feature in enumerate(user_input_features):
    with [col1, col2, col3][idx % 3]:
        input_data[feature] = st.number_input(
            label=feature.replace('_', ' ').title(),
            min_value=0.0,
            value=1.0,
            step=0.1
        )

# Create the input_df with all expected features, filling missing with 0.0
full_input = {feature: input_data.get(feature, 0.0) for feature in expected_features}
input_df = pd.DataFrame([full_input])[expected_features]

if st.button('Predict'):
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.error(f"🔴 **Prediction: Malignant Tumor** (Probability: {probability:.2%})")
    else:
        st.success(f"🟢 **Prediction: Benign Tumor** (Probability: {probability:.2%})")

