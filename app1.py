# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 14:54:37 2020
@author: Mushtaq Ali
"""


######################
# Import libraries
######################
import streamlit as st
import pickle
import pubchempy as pcp
from PIL import Image
import pandas as pd
from rdkit import Chem
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
import requests
from bs4 import BeautifulSoup



######################
# Custom function
######################
## Calculate molecular descriptors
def getAromaticProportion(m):
    aromatic_list = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
    aromatic = 0
    for i in aromatic_list:
        if i:
            aromatic += 1
    heavy_atom = Lipinski.HeavyAtomCount(m)
    return aromatic / heavy_atom if heavy_atom != 0 else 0


def generate(smiles):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData = np.arange(1, 1)
    i = 0
    for mol in moldata:

        desc_MolLogP = Crippen.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Lipinski.NumRotatableBonds(mol)
        desc_AromaticProportion = getAromaticProportion(mol)
        desc_Ringcount        =   Descriptors.RingCount(mol)
        desc_TPSA = Descriptors.TPSA(mol)
        desc_Hdonrs=Lipinski.NumHDonors(mol)
        desc_SaturatedRings = Lipinski.NumSaturatedRings(mol)   
        desc_AliphaticRings = Lipinski.NumAliphaticRings(mol) 
        desc_HAcceptors  =     Lipinski.NumHAcceptors(mol)
        desc_Heteroatoms =    Lipinski.NumHeteroatoms(mol)
        desc_Max_Partial_Charge =  Descriptors.MaxPartialCharge(mol)
        desc_FP_density =  Descriptors.FpDensityMorgan1(mol)
        desc_num_valence_electrons = Descriptors.NumValenceElectrons(mol)
        NHOH_count = Lipinski.NHOHCount(mol)
        SP3_frac = Lipinski.FractionCSP3(mol)
        SP_bonds = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[^1]')))
        #Ipc      = Descriptors.Ipc(mol)
        #HallKierAlpha= Descriptors.HallKierAlpha(mol)
        #Labute_ASA = Descriptors.LabuteASA(mol)



        #desc_molMR=Descriptors.MolMR(mol)
        row = np.array([desc_MolLogP,
                        desc_MolWt, desc_NumRotatableBonds,
                        desc_AromaticProportion,desc_Ringcount,desc_TPSA,desc_Hdonrs,desc_SaturatedRings,desc_AliphaticRings,
                        desc_HAcceptors,desc_Heteroatoms,
                        desc_Max_Partial_Charge,desc_num_valence_electrons,desc_FP_density,NHOH_count,SP3_frac,SP_bonds])
                            #,Ipc,HallKierAlpha,Labute_ASA])#,desc_num_valence_electrons])

        if i == 0:
            baseData = row
        else:
            baseData = np.vstack([baseData, row])
        i = i + 1

    columnNames = ["MolP","MolWt", 
                   "NumRotatableBonds", "AromaticProportion"
                  ,"Ring_Count","TPSA","H_donors", "Saturated_Rings","AliphaticRings","H_Acceptors","Heteroatoms","Max_Partial_Charge",
                  "valence_electrons","FP_density","NHOH_count","SP3_frac","SP_bonds"]
                  #,"Ipc","HallKierAlpha","Labute_ASA"]
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)
    
    return descriptors
def remove_invalid(smiles):
    """
    Removes invalid molecules from the dataset.
    """
    valid = [sm for sm in smiles if MolFromSmiles(sm)]
    if len(valid) == len(smiles):
        return smiles, "All provided SMILES are valid!"
    return valid, "Some SMILES are invalid! Showing results for valid SMILES only!"
######################
# Page Title
######################


st.set_page_config(page_title="AqSolPred: Online Solubility Prediction Tool",layout="wide")
		   
st.write("""# Solibility Prediction on Aqueous Solvent """)

image = Image.open('sol_image.jpeg')
st.image(image, use_column_width=False)
#st.set_wide_mode()
#st.set_page_config(layout="wide")
#st.set_option('wideMode' , True)
#st.set_page_config(
  #   page_title="SOlubility Prediction App",
  #   page_icon="🧊",
  #   layout="wide",
  #   initial_sidebar_state="expanded",
  #   menu_items={
   #      'Get Help': 'https://www.extremelycoolapp.com/help',
   #      'Report a bug': "https://www.extremelycoolapp.com/bug",
   #      'About': "# This is a header. This is an SOlubility Prediction App!"
    # }
 #)


######################
# Input molecules (Side Panel)
######################

st.sidebar.write('**Type SMILES below, In one line one smiles (At least enter two smiles)**')

## Read SMILES input
SMILES_input = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C\nCC(=O)OC1=CC=CC=C1C(=O)O"

SMILES = st.sidebar.text_area('then press ctrl+enter', SMILES_input, height=17,max_chars=2000)
SMILES = SMILES.split('\n')
SMILES, msg = remove_invalid(SMILES)
st.sidebar.write(msg)
#smiles = list(filter(None, SMILES))


st.sidebar.write("""---------**OR**---------""")
st.sidebar.write("""**Upload a 'csv' file with a column named 'SMILES'** (Max:2000)""")

   
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # data
    SMILES=data["SMILESs"]  


# st.header('Input SMILES')
# SMILES[1:] # Skips the dummy first item

# Use only top 300
if len(SMILES)>2000:
    SMILES=SMILES[0:2000]
	
## Calculate molecular descriptors
generated_descriptors = generate(SMILES)

#Import pretrained models
#mlp_model_import = pickle.load(open('models/model_reg_94.pkl', 'rb'))
trained_model= xgb.Booster()
trained_model.load_model('models/model_xgb_95 2.bin')
#rf_model_import = pickle.load(open('models/model_rf_93.pkl', 'rb'))
mols = [Chem.rdmolfiles.MolFromSmiles(SMILES_string) for SMILES_string in SMILES]
#Convert training molecules into training fingerprints
bi = {}
fingerprints = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, bitInfo= bi, nBits=512) for m in mols]

#Convert training fingerprints into binary, and put all training binaries into arrays
import numpy as np 

fingerprints_array = []
for fingerprint in fingerprints:
        array = np.zeros((1,), dtype= int)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        fingerprints_array.append(array)

fingerprints_array=pd.DataFrame(fingerprints_array)
df1=pd.concat([fingerprints_array,generated_descriptors],axis=1)
#print(df1)
def smiles_to_sol(SMILES):
    prop=pcp.get_properties([ 'MolecularWeight'], SMILES, 'smiles')
    x = list(map(lambda x: x["CID"], prop))
    y=x[0]
   #print(y)
    x = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/%s/xml"
    data=requests.get(x % y)
    #print(data)
    html = BeautifulSoup(data.content, "xml")
    solubility = html.find(name='TOCHeading', string='Solubility')
    if solubility ==None:
      return None
    else:
      solub=solubility.find_next_sibling('Information').find(name='String').string
      return solub




#predict test data (MLP,XGB,RF)
#pred_mlp = mlp_model_import.predict(df1)   
df3 = xgb.DMatrix(df1)
pred_xgb = trained_model.predict(df3)
pred_xgb = pred_xgb-0.18
#pred_rf = rf_model_import.predict(df1)
mol_liter =10**pred_xgb
#mol = Chem.MolFromSmiles(SMILES)
#MolWt = Chem.Descriptors.MolWt(mol)
 
#mol_list = []
#for smiles in SMILES:
 #   mol = Chem.MolFromSmiles(smiles)
  #  mol_list.append(mol)
#MolWt = Chem.Descriptors.MolWt(mol_list)
MolWt = generated_descriptors["MolWt"]
Gram_liter=(10**pred_xgb)*MolWt
P_sol=[] ## List where I am storing the solubility data 
for i in range(len(SMILES)): ## Dataframe which has SMILES 
  try:
    #time.sleep(1) # Sleep for 3 seconds
    sol=smiles_to_sol(SMILES[i]) ### Function to get the data from PubChem 
    P_sol.append(sol)
  except AttributeError as e:
    sol=='No string'
    P_sol.append(sol)
#calculate consensus
#pred_consensus=(pred_mlp+pred_xgb+pred_rf)/3
# predefined_models.get_errors(test_logS_list,pred_enseble)

# results=np.column_stack([pred_mlp,pred_xgb,pred_rf,pred_consensus])
df_results = pd.DataFrame(SMILES, columns=['SMILES'])
df_results["Predicted - LogS"]=pred_xgb
df_results=df_results.round(3)
df_results["Mol/Liter"]=mol_liter
df_results["Gram/Liter"]=Gram_liter
df_results["Experiment Solubility-PubChem"]=P_sol

    #df
# df_results.to_csv("results/predicted-"+test_data_name+".csv",index=False)

st.header('Predicted LogS values')
#df_results # Skips the dummy first item
df_results
# download=st.button('Download Results File')
# if download:
csv = df_results.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()  # some strings
linko= f'<a href="data:file/csv;base64,{b64}" download="aqsolpred_predictions.csv">Download csv file</a>'
st.markdown(linko, unsafe_allow_html=True)
 
st.header('Computed molecular descriptors')
generated_descriptors # Skips the dummy first item


st.write("""
# About AqSolPred
Sol Prediction  is an  accurate solubility prediction model that consists consensus of 3 ML algorithms (Neural Nets, Random Forest, and XGBoost). 
AqSolPred is developed using a quality-oriented data selection method described in [1] and trained on AqSolDB [2] largest publicly available aqueous solubility dataset.
AqSolPred showed a top-performance (R2 - 0.95 )

If you are using the predictions from Sol Pred on your work, 
 please cite these papers: [1, 2]

[1] Sorkun, M. C., Koelman, J.M.V.A. & Er, S. (2021). [Pushing the limits of solubility prediction via quality-oriented data selection](https://www.cell.com/iscience/fulltext/S2589-0042(20)31158-5), iScience, 24(1), 101961.

[2] Sorkun, M. C., Khetan, A., & Er, S. (2019).  [AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds](https://www.nature.com/articles/s41597-019-0151-1). Scientific data, 6(1), 1-8.

[3] Huuskonen, J. Estimation of aqueous solubility for a diverse set of organic compounds based on molecular topology. Journal of Chemical Informationand Computer Sciences 40, 773–777 (2000).
Special thanks: 

This web app is developed based on the tutorials and the template of [DataProfessor's repository](https://github.com/dataprofessor/code/tree/master/streamlit/part7). 
                                                                                         
**Contact:** [Mushtaq Ali](www.linkedin.com/in/mushtaq-ali/)
""")



