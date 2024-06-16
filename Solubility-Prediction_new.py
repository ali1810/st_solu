import streamlit as st
from streamlit_option_menu import option_menu

import home,project,contact

import streamlit as st


sideb = st.sidebar
check1 = sideb.button("infrared f group prediction")
#textbyuser = st.text_input("Enter some text")
if check1:
    #st.info("Code is analyzing your text.")
    st.write("Upload a file for functional group Prediction")
    uploaded_file = st.file_uploader("Choose a file")      
if st.button('Prediction for input file'):
  st.write('Enter your function')    
    
