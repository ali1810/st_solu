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
                    sol= None
#sol.append(solub)
              else:
                solub=solubility.find_next_sibling('Information').find(name='String').string
                sol= solub
                st.write("Work is in progress...cccc::", sol)
              mol1 = Chem.MolFromSmiles(SMILES)
    
               # calculate the log octanol/water partition descriptor
              single_MolLogP = Descriptors.MolLogP(mol1)
    
              # calculate the molecular weight descriptor
              single_MolWt   = Descriptors.MolWt(mol1)
    
             # calculate of the number of rotatable bonds descriptor
              single_NumRotatableBonds = Descriptors.NumRotatableBonds(mol1)
              aromatic_list = [mol1.GetAtomWithIdx(i).GetIsAromatic() for i in range(mol1.GetNumAtoms())]
              aromatic = 0
              for i in aromatic_list:
                if i:
                 aromatic += 1
                 heavy_atom = Lipinski.HeavyAtomCount(mol1)
              #return aromatic / heavy_atom if heavy_atom != 0 else 0
             # calculate the aromatic proportion descriptor
              single_AP = aromatic / heavy_atom if heavy_atom != 0 else 0

             # Calculate ring count 
              single_RC= Descriptors.RingCount(mol1)

             # Calculate TPSA 
              single_TPSA=Descriptors.TPSA(mol1)

             # Calculate H Donors  
              single_Hdonors=Lipinski.NumHDonors(mol1)

            # Calculate saturated Rings 
              single_SR= Lipinski.NumSaturatedRings(mol1) 

            # Calculate Aliphatic rings 
              single_AR =Lipinski.NumAliphaticRings(mol1)
                  # Calculate Hydrogen Acceptors 
              single_HA = Lipinski.NumHAcceptors(mol1)

              # Calculate Heteroatoms
              single_Heter = Lipinski.NumHeteroatoms(mol1)

              
              single_Max_Partial_Charge =  Descriptors.MaxPartialCharge(mol1)
              single_FP_density =  Descriptors.FpDensityMorgan1(mol1)
              single_num_valence_electrons = Descriptors.NumValenceElectrons(mol1)
              single_NHOH_count = Lipinski.NHOHCount(mol1)
              single_SP3_frac = Lipinski.FractionCSP3(mol1)
              single_SP_bonds = len(mol1.GetSubstructMatches(Chem.MolFromSmarts('[^1]')))
            
                        
    

              # put the descriptors in a list
              rows = np.array([single_MolLogP, single_MolWt, single_NumRotatableBonds, single_AP,single_RC,
                       single_TPSA,single_Hdonors,single_SR,single_AR,
                      single_HA,single_Heter,single_Max_Partial_Charge,single_FP_density,single_num_valence_electrons,
                     single_NHOH_count,single_SP3_frac,single_SP_bonds])
    
             # add the list to a pandas dataframe
             #single_df = pd.DataFrame(single_list).T
              baseData = np.vstack([rows])
              # rename the header columns of the dataframe
    
                #columnNames = ["MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion","Ring_Count","TPSA","H_donors","Saturated_Rings","AliphaticRings","H_Acceptors","Heteroatoms"]
              columnNames = ["MolP","MolWt", 
                   "NumRotatableBonds", "AromaticProportion"
                  ,"Ring_Count","TPSA","H_donors", "Saturated_Rings","AliphaticRings","H_Acceptors","Heteroatoms","Max_Partial_Charge",
                  "valence_electrons","FP_density","NHOH_count","SP3_frac","SP_bonds"]
 
              descriptors1 = pd.DataFrame(data=baseData, columns=columnNames)
              #generated_descriptors1= predictSingle(smiles)
              mol = Chem.MolFromSmiles(SMILES)
              fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
              arr = np.zeros((0,), dtype=np.int8)
              arr1=DataStructs.ConvertToNumpyArray(fp,arr)
              arr2 =pd.DataFrame(arr)
              array = arr2.T
    #print(array.shape)
#print(fingerprints_array1)
#mols = [Chem.rdmolfiles.MolFromSmiles(SMILES_string) for SMILES_string in smiles]
              trained_model= xgb.Booster()
              trained_model.load_model('model_xgb_95 2.bin')
              df3=pd.concat([array,descriptors1],axis=1)
              df3 = xgb.DMatrix(df3)
	
#print(df3)
              pred_rf1 = trained_model.predict(df3)
              #pred_rf1 =  (pred_rf1-0.30)
              pred_rf2 =  np.round(pred_rf1,2)	
              mol_liter1   =  10**pred_rf1
              mol_liter2   = np.round(mol_liter1,2)

    	        #compounds = pcp.get_compounds(SMILES, namespace='smiles')
              #match = compounds[0]
              #c_name =match.iupac_name

              MolWt1     =descriptors1["MolWt"]

              Gram_liter1  =(10**pred_rf1)*MolWt1
              #Gram_liter1 = round(Gram_liter1,2) 	
               #P_sol1 =smiles_to_sol(smiles) ## calling function to get the solubility from <pubchem
#df_results = pd.DataFrame(df_results1)
    #render_mol(blk)
              data = dict(SMILES=SMILES, Predicted_LogS=pred_rf2, 
              Mol_Liter=mol_liter2,Gram_Liter=Gram_liter1,Experiment_Solubility_PubChem=sol)
              df = pd.DataFrame(data, index=[0])

              st.write('Predicted LogS values for single smiles')
              st.table(df.style.format({"Predicted_LogS": "{:.2f}","Mol_Liter":"{:.2f}","Gram_Liter":"{:.2f}"}))
              st.write('Computed molecular descriptors')
              df1 = pd.DataFrame(descriptors1, index=[0])
              #descriptors1 # Skips the dummy first item
              st.table(df1)

    
  

        


	      
	      
	      
	  

	      
        # Add more content to the container dynamically
	    #page1()
	      
