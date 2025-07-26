import streamlit as st
import joblib

model=joblib.load('model .pkl')
vec=joblib.load('vectorizer.pkl')

st.title('Disaster Tweet Classifier')

user_input=st.text_input('Enter a tweet')

if st.button('Predict'):
    if user_input.strip()=="":
        st.warning("Please enter a tweet")
    else:
        input_vec=vec.transform([user_input])
        pred=model.predict(input_vec)
        label='Disaster' if pred[0]==1 else "Not Disaster"
        st.success(f"Prediction: {label}")