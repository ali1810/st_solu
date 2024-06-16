import streamlit as st 
from PIL import Image


    #st.title("Page 2")
    #st.write("This is the page for project details")
image = Image.open('flow7.jpeg')
	    
col1, mid, col2 = st.columns([40,0.5,0.5])
with col1:
        st.image(image,use_column_width=False)
with col2:
            #showmol(xyzview,height=300,width=400)
	    st.write("")           
