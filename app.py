import pandas as pd  # not pandas as pandas
import pickle as pk
import streamlit as st

# Load model
model = pk.load(open('/home/user/Desktop/ML_Projects/House_prediction_model.pk1', 'rb'))

# Set header
st.header('Banglore House Price Predictor')  # use function, not assignment

# Load data
data = pd.read_csv('/home/user/Desktop/ML_Projects/Cleaned_data.csv')

# Input fields
loc = st.text_input('Choose the location', data['location'].iloc[0])
sqft = st.number_input('Enter Total sqft')
beds = st.number_input('Enter No of Bedrooms')
bath = st.number_input('Enter No of Bathrooms')
balc = st.number_input('Enter No of Balconies')

# Prepare input data
input_df = pd.DataFrame([[loc, sqft, bath, balc, beds]], columns=['location', 'total_sqft', 'bath', 'balcony', 'bedrooms'])

# Prediction
if st.button('Predict Price'):
    output = model.predict(input_df)
    st.success('Price of the House is ₹{:,.0f}'.format(output[0] * 100000))
