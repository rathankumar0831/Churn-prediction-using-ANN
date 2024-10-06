import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle



## Load the Trained model, scaler and onhot file
model = tf.keras.models.load_model('model.h5')


with open('onehotencode_geo.pkl', 'rb') as file:
    onehotencode_geo=pickle.load(file)

with open('label_Encoder_gender.pkl', 'rb') as file:
    label_Encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## streamlit app title
st.title("Customer churn Prediction")

# user input
geography = st.selectbox('Geography', onehotencode_geo.categories_[0])
gender = st.selectbox('Gender', label_Encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


#prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_Encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary' : [estimated_salary]
  })

geo_encoder = onehotencode_geo.transform([[geography]]).toarray()
geo_encoder_df = pd.DataFrame(geo_encoder, columns=onehotencode_geo.get_feature_names_out(['Geography']))

## concatination
input_data = pd.concat([input_data.reset_index(drop = True), geo_encoder_df], axis=1)


## scaler the input
input_data_scaled = scaler.transform(input_data)


# prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'churn probability : {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('the customer is likely to churn')
else:
    st.write('the customer is not likely to churn')












