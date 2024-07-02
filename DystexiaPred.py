import streamlit as st
import pandas as pd
import joblib


# Load the trained model
model = joblib.load('C:/Users/uchen/PycharmProjects/Dyslexia/meta_model.pkl')

# Title and description
st.title('Dyslexia Prediction App')
st.write('This app predicts the likelihood of Dyslexia based on several input features.')

# Input fields
gender = st.selectbox('Gender', [1, 0])
native_language = st.selectbox('Native Language', [1,0])
other_language = st.selectbox('Other Language', [1,0])
age = st.number_input('Age', min_value=1, max_value=17)
clicks = st.number_input('Clicks', min_value=0)
hits = st.number_input('Hits', min_value=0)
misses = st.number_input('Misses', min_value=0)
scores = st.number_input('Scores', min_value=0)
accuracy = st.number_input('Accuracy', min_value=0.0, max_value=1.0, step=0.01)
missrate = st.number_input('Missrate', min_value=0.0, max_value=1.0, step=0.01)

# Prediction
if st.button('Predict'):
    input_data = pd.DataFrame({
        'Gender': [gender],
        'NativeLanguage': [native_language],
        'OtherLanguage': [other_language],
        'Age': [age],
        'Clicks': [clicks],
        'Hits': [hits],
        'Misses': [misses],
        'Scores': [scores],
        'Accuracy': [accuracy],
        'Missrate': [missrate]
    })

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Display result
    st.write(f'Prediction: {"Yes, this student is Dyslexic." if prediction[0] else "No, this student is not Dyslexic."}')
    st.write(f'Prediction Probability: {prediction_proba[0][1]:.2f}')

# Run the app
if __name__ == '__main__':
    st.write('Welcome to the Dyslexia Prediction App. Please input the required fields and press "Predict".')
