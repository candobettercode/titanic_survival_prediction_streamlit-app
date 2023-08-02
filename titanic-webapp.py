# -*- coding: utf-8 -*-
"""
Created on Jul 23 2023

@author: siddhesh
"""

import numpy as np
import pickle
import streamlit as st
import sklearn

st.set_page_config(page_title="Titanic Prediction!!!", page_icon=":ship:", layout="centered")
# giving a title
st.title(":ocean: Titanic ML Survival Prediction")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)

st.text("Tecknowcode @Tecknowcode")
st.text("Siddhesh M.(candobettercode)")

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
      return 'The person is not survived :broken_heart: :cry:'
    else:
      return 'The person is survived :sparkling_heart: :sunglasses:'
  
    
  
def main():
        
    st.subheader("Select features:")
    # getting the input data from the user
    
    Sex = st.radio('Enter Gender of Passenger',['male','female'],horizontal=True)
    
    col1,col2 = st.columns(2)
  
    with col1:
       Pclass = st.radio('Select Ticket Class',[1,2,3])
       st.caption("1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class", unsafe_allow_html=False)        
       Age = st.number_input("Enter Age of Passenger",1,99)
       Parched = st.number_input('Enter number of parents/ children aboard the Titanic',1,10)

    with col2:
       Embarked = st.radio('Select Port of Embarkation',['S','C','Q'])
       st.caption("C = Cherbourg, Q = Queenstown, S = Southampton", unsafe_allow_html=False)
       Fare = st.text_input('Enter Fare')
       Sibsp = st.number_input('Enter number of siblings/ spouses aboard the Titanic',1,10)
               
    
    selected_options = {'Gender':Sex,
                        'Passenger Class':Pclass,
                        'Age':Age,
                        'Parents/ Children on Boat':Parched,
                        'Port of Embarkation':Embarked,
                        'Fare':Fare,
                        'Siblings/ Spouse on Boat':Sibsp}
    
    st.write(selected_options)

    # code for Prediction
    prediction = ''
    
    # creating a button for Prediction
    
    if st.button('Survival Test Result'):
        prediction = survival_prediction([Pclass,Sex,Age,Sibsp,Parched,Fare,Embarked])
        
        
    st.success(prediction)
    
       
    
    
if __name__ == '__main__':
    main()
    
    
