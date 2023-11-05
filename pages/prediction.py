# import streamlit as st
# import pickle
# import numpy as np

# st.title("Another Model Prediction")

# # Load the second pre-trained model using pickle
# with open('C:/Users/Aditee/OneDrive/Documents/GitHub/Hacksc_1/random_forest_model.pkl', 'rb') as model_file:
#     another_model = pickle.load(model_file)

# # Create input fields for user to enter data specific to the second model
# # radius_mean = st.number_input("Radius Mean", min_value=0.0, value=123456.0)
# # texture_mean = st.number_input("Texture Mean", min_value=0.0, value=7.0)
# # perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, value=8.0)
# # area_mean = st.number_input("Area Mean", min_value=0.0, value=9.0)
# # smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, value=10.0)
# # compactness_mean = st.number_input("Compactness Mean", min_value=0.0, value=11.0)
# # concavity_mean = st.number_input("Concavity Mean", min_value=0.0, value=12.0)
# # concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, value=13.0)
# # symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, value=14.0)
# # fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, value=15.0)

# # Define input fields for the remaining parameters
# # Repeat the process for all the parameters you listed

# # Process the user input and make predictions specific to the second model
# if st.button("Predict (Another Model)"):
#     input_data = np.array([
#         radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
#         compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean
#         # Add input values for the remaining parameters
#     ]).reshape(1, -1)

#     prediction = another_model.predict(input_data)

#     # Display the prediction result based on the second model
#     if prediction[0] == 1:
#         st.success("The prediction from the second model is positive.")
#     else:
#         st.error("The prediction from the second model is negative.")





import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression

# Load the trained model
# model = pickle.load('/Users/Utkarsha_1/Documents/heyy/svm.pkl')
import pickle

# Open the file in binary read mode
with open('C:/Users/Aditee/OneDrive/Documents/GitHub/Hacksc_1/svm.pkl', 'rb') as file:
    model = pickle.load(file)


# Define the data preprocessing function
def preprocess_data(radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
    compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
    fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
    smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se,
    fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst,
    smoothness_worst, compactness_worst,concavity_worst, concave_points_worst,
    symmetry_worst, fractal_dimension_worst):
    # Convert categorical variables to numericals representations
    # menopause_mapping = {'premeno': 0, 'ge40': 1, 'lt40': 2}
    # irradiate_mapping = {'no': 0, 'yes': 1}
    # node_caps_mapping = {'no': 0, 'yes': 1}
    # menopause = menopause_mapping[menopause]
    # irradiate = irradiate_mapping[irradiate]
    # node_caps = node_caps_mapping[node_caps]
    
    

    # Standardize numerical variables
    data = np.array([radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
    compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
    fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
    smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se,
    fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst,
    smoothness_worst, compactness_worst,concavity_worst, concave_points_worst,
    symmetry_worst, fractal_dimension_worst]).reshape(1, -1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    return data

# Create the Streamlit app
st.title('Breast Cancer Reccurence Prediction')

# Collect user input for the breast cancer parameters
radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
    compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
    fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
    smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se,
    fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst,
    smoothness_worst, compactness_worst,concavity_worst, concave_points_worst,
    symmetry_worst, fractal_dimension_worst


if st.button('Submit'):
    # Preprocess the user input data
    data = preprocess_data(age, menopause, tumer_size, inv_nodes, node_caps, irradiate)

    # Make a prediction using the trained model
    prediction = model.predict(data)

    # Display the prediction result
    if prediction == 0:
        result = "No Recurrence"
    else:
        result = "Recurrence"

    # Show the prediction result
    st.write(f'Prediction: {result}')


