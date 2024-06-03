import streamlit as st
import requests 
import pandas as pd

st.title('Titanic prediction')

# put training button 
p = st.button("Predict")
if p:
    response=requests.post('http://backend:8000/predict',json=input_path)
    st.write('The prediction is in the following path')
    st.write(response.json())
    predictions = pd.read_csv(response.json()["outupt path"])
    st.write(predictions)
    st.image(predictions.hist())

p2 = st.button("Do data analysis")
if p2:
    # this is for data analysis endpoint 
    response2 = requests.post('http://backend:8000/analysis',json=input_path)
    
    # load images from data visualisation part and dataframes from the data analysis 
    # st.image(...)
    # 