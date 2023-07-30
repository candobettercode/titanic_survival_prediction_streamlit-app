# -*- coding: utf-8 -*-
"""
Created on Jul 23 2023

@author: siddhesh
"""

import numpy as np
import pickle
import streamlit as st
import sklearn


# loading the saved model
loaded_model = pickle.load(open('titanic_model/pipe.pkl','rb' ))


# creating a function for Prediction

def survival_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not survived'
    else:
      return 'The person is survived'
  
    
  
def main():
    
    
    # giving a title
    st.title('Titanic Passengers Survival Prediction Web App')
    st.text('by Siddhesh M.')
    
    
    # getting the input data from the user
    
    
    Pclass = st.radio('Value of PClass',[1,2,3])
    Sex = st.radio('Enter Gender',['male','female'])
    Age = st.text_input('Age')
    Sibsp = st.text_input('Sibsp')
    Parched = st.text_input('Parched')
    Fare = st.text_input('Enter Fare')
    Embarked = st.radio('Select Embarked',['S','C','Q'])
        
    
    # code for Prediction
    prediction = ''
    
    # creating a button for Prediction
    
    if st.button('Survival Test Result'):
        prediction = survival_prediction([Pclass,Sex,Age,Sibsp,Parched,Fare,Embarked])
        
        
    st.success(prediction)
    
       
    
    
if __name__ == '__main__':
    main()
    