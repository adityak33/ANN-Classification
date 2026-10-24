import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

## Load the training model

model = tf.keras.models.load_model("model.h5")

## Load the encoders and scalers
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("one_hot_encoder_geo.pkl", "rb") as file:
    one_hot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)    


## Streamlet app
st.title("Customer Churn Prediction")
st.caption("Enter the customer's details to estimate churn probability.")

# User input
geography = st.selectbox("Geography", one_hot_encoder_geo.categories_[0])
gender = st.selectbox("Gender",label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance", min_value=0.0, max_value=250000.0, value=100000.0, step=1000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=100000.0, step=1000.0)
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])


# Prepare the input data
input_data = pd.DataFrame({
    "CreditScore" : [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age" : [age],
    "Tenure" : [tenure],
    "Balance" : [balance],
    "NumOfProducts" : [num_of_products],
    "HasCrCard" : [has_cr_card],
    "IsActiveMember" : [is_active_member],
    "EstimatedSalary" : [estimated_salary]
})

# One-hot encode Geography, using the encoder fitted during training.
geo_encoded = one_hot_encoder_geo.transform(pd.DataFrame({"Geography": [geography]})).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=one_hot_encoder_geo.get_feature_names_out(["Geography"]),
)

# Cobine one-hot encoded columns with input data
# Keep the exact training feature order, then apply the same fitted scaler used
# to train the ANN. The model was trained on scaled features, not raw values.
model_input = pd.concat([input_data, geo_encoded_df], axis=1)
if hasattr(scaler, "feature_names_in_"):
    model_input = model_input.loc[:, scaler.feature_names_in_]
input_data_scaled = scaler.transform(model_input)

# Prediction Churn
prediction = model.predict(input_data_scaled, verbose=0)
prediction_proba = prediction[0][0]

st.write(f"Churn Probability: {prediction_proba:.1%}")

if prediction_proba > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")    
