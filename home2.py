import streamlit as st
from PIL import Image
from stmol import showmol
import py3Dmol
import re
import numpy as np 
from rdkit.Chem import AllChem
import pubchempy as pcp
import streamlit as st
import pickle
from PIL import Image
import pandas as pd
from rdkit import Chem
#from rdkit.Chem import Draw 
import xgboost
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import base64
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem import MolFromSmiles, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen
import streamlit as st
import base64
from streamlit_shap import st_shap
import shap
from xgboost import XGBRegressor
import xgboost as xgb
from urllib.request import urlopen
from PIL import Image
#from flask.wrappers import Request
#import threading
import requests
import streamlit as st
from bs4 import BeautifulSoup


def app():
	
      st.write('**Type SMILES below**')
      SMILES = st.text_input('then press predict button', value ="CC(=O)OC1=CC=CC=C1C(=O)O")
#style = st.selectbox('Chemical structure',['stick','ball-and-stick','line','cross','sphere'])
#spin = st.checkbox('Spin', value = False)	
      #col1, col2, col3 = st.columns([10,2,11.5])
      #with col1:	
	#  st.write("   2 D Structure of the smiles  ")
      #with col2:
	#  st.write("")
      #with col3:
       #   st.write(" 3 D Structure  of the smiles")
 #       st.write("""Use mouse pointer to rotate the structure""")

      prop=pcp.get_properties([ 'MolecularWeight'], SMILES, 'smiles')
      x = list(map(lambda x: x["CID"], prop)) 
      y=x[0]
    #print(y) 
      x = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/%s/PNG?image_size=300x300"
      url=(x % y)
#print(url)
      img = Image.open(urlopen(url))
      mol = Chem.MolFromSmiles(SMILES)
      mol = Chem.AddHs(mol)
      AllChem.EmbedMolecule(mol)
      mblock = Chem.MolToMolBlock(mol)
      xyzview = py3Dmol.view(width=300,height=300)
    #xyzview = py3Dmol.view(query=′pdb:1A2C′)
      xyzview.addModel(mblock,'mol')
      xyzview.setStyle({'model': -1}, {"cartoon": {'color': 'spectrum'}})
      style = 'stick'
#bcolor = st.sidebar.color_picker('Pick background Color', '#0C0C0B')
#xyzview.spin(True)
#if spin:
 #   xyzview.spin(True)
#else:
 #    xyzview.spin(False)
    #xyzview.setStyle({'sphere':{}})
      xyzview.setBackgroundColor('#EAE5E5')
      xyzview.zoomTo()
      xyzview.setStyle({style:{'color':'spectrum'}})
    


      col1, mid, col2 = st.columns([15,2.5,15])
#col1, mid, col2 = st.columns([15,0.5,15])
      with col1:
          st.image(img, use_column_width=False)
      with col2:
          showmol(xyzview,height=300,width=400) 
      #if st.button("predict"):
     # if st.button("Predict"):
	#      st.write("This is some content that should remain on the page.")        	      
          #page1()
      #        st.write("work in Progress") 
      if st.button("Predict"):
	      #prop=pcp.get_properties([ 'MolecularWeight'], SMILES, 'smiles')
              x = list(map(lambda x: x["CID"], prop))
              y=x[0]
              x = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/%s/xml"
              data=requests.get(x % y)
    
              html = BeautifulSoup(data.content, "xml")
              solubility = html.find(name='TOCHeading', string='Solubility')
              if solubility ==None:
                    Sol= None
#sol.append(solub)
              else:
                solub=solubility.find_next_sibling('Information').find(name='String').string
                 Sol= solub


	      
	      
	      
	  

	      
        # Add more content to the container dynamically
	    #page1()
	      st.write("Work is in progress...cccc",Sol)
  
