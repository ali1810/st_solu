# -*- coding: utf-8 -*-
"""
Created on Sun August 12 14:54:37 2022
@author: Mushtaq Ali
"""


######################
# Import libraries
######################
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

from bs4 import BeautifulSoup


######################
# Custom function. ...
######################
CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"
### Function to get the compound name ...
def smiles_to_iupac(smiles):
    rep = "iupac_name"
    url = CACTUS.format(smiles, rep)
    response = requests.get(url)
    response.raise_for_status()
    return response.text
def smiles_iupac(sm):
#smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    compounds = pcp.get_compounds(sm, namespace='smiles')
  #print(compounds)
    match = compounds[0]
    return match.iupac_name


def makeblock(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    mblock = Chem.MolToMolBlock(mol)
    return mblock

def render_mol(xyz):
    xyzview = py3Dmol.view(width=400,height=300)
    #xyzview = py3Dmol.view(query=′pdb:1A2C′)
    xyzview.addModel(xyz,'mol')
    xyzview.setStyle({'model': -1}, {"cartoon": {'color': 'spectrum'}})
    #bcolor = st.sidebar.color_picker('Pick background Color', '#0C0C0B')
    style = st.sidebar.selectbox('Chemical structure',['stick','ball-and-stick','line','cross','sphere'])
#spin = st.sidebar.checkbox('Spin', value = False)
    spin = st.sidebar.checkbox('Animation', value = True)
    xyzview.spin(True)
    if spin:
      xyzview.spin(True)
    else:
      xyzview.spin(False)
    #xyzview.setStyle({'sphere':{}})
    xyzview.setBackgroundColor('#EAE5E5')
    xyzview.zoomTo()
    xyzview.setStyle({style:{'color':'spectrum'}})
    showmol(xyzview,height=300,width=400) 


## Calculate molecular descriptors
def smiles_to_sol(SMILES):
    prop=pcp.get_properties([ 'MolecularWeight'], SMILES, 'smiles')
    x = list(map(lambda x: x["CID"], prop))
    y=x[0]
   #print(y)
    x = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/%s/xml"
    data=requests.get(x % y)
    print(data)
    html = BeautifulSoup(data.content, "xml")
    solubility = html.find(name='TOCHeading', string='Solubility')
    if solubility ==None:
      return None
#sol.append(solub)
    else:
      solub=solubility.find_next_sibling('Information').find(name='String').string
      return solub
def smiles_to_img(SMILES):
    prop=pcp.get_properties([ 'MolecularWeight'], SMILES, 'smiles')
    x = list(map(lambda x: x["CID"], prop))
    y=x[0]
    #print(y)
    x = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/%s/PNG?image_size=400x300"
    url=(x % y)
#print(url)
    img = Image.open(urlopen(url))
    return img 

def getAromaticProportion(m):
    aromatic_list = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
    aromatic = 0
    for i in aromatic_list:
        if i:
            aromatic += 1
    heavy_atom = Lipinski.HeavyAtomCount(m)
    return aromatic / heavy_atom if heavy_atom != 0 else 0
def predictSingle(SMILES):
    """
    This function predicts the four molecular descriptors: the octanol/water partition coefficient (LogP),
    the molecular weight (Mw), the number of rotatable bonds (NRb), and the aromatic proportion (AP) 
    for a single molecule
    
    The input arguments are SMILES molecular structure and the trained model, respectively.
    """
    
    # define the rdkit moleculer object
    mol1 = Chem.MolFromSmiles(SMILES)
    
    # calculate the log octanol/water partition descriptor
    single_MolLogP = Descriptors.MolLogP(mol1)
    
    # calculate the molecular weight descriptor
    single_MolWt   = Descriptors.MolWt(mol1)
    
    # calculate of the number of rotatable bonds descriptor
    single_NumRotatableBonds = Descriptors.NumRotatableBonds(mol1)
    
    # calculate the aromatic proportion descriptor
    single_AP = getAromaticProportion(mol1)

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
    return descriptors1 
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
    valid = Chem.MolFromSmiles(smiles)
    if len(valid) == len(smiles):
        return smiles, "Given  SMILES is valid!"
    return valid, "SMILES is invalid! Showing results for valid SMILES only!"
######################
# Page Title
######################
#st.set_page_config(page_title="AqSolPred: Online Solubility Prediction Tool")
st.set_page_config(page_title="AqSolPred: Online Solubility Prediction Tool",layout="wide")
st.write("""# Solibility Prediction on Aqueous Solvent """)
image = Image.open('Flow.jpeg')
col1, col2, col3 = st.columns([0.5,2.0,0.5])
with col1:
	st.write("")

with col2:
	
        st.image(image, use_column_width=6)
with col3:	
        st.write("")
	
	
	
       #st.header("    2D and 3D Structure of the smiles      ")
#st.image("https://i.imgflip.com/amucx.jpg")
        #st.write("")

col1, col2, col3 = st.columns([10,2,11.5])

with col1:
	st.header("   2 D Structure of the smiles  ")

with col2:
	st.write("")
with col3:
        st.header(" 3 D Structure  of the smiles")
        st.write("""Use mouse pointer to rotate the structure""")

######################
# Input molecules (Side Panel)
######################

st.sidebar.write('**Type SMILES below**')

## Read SMILES input
#SMILES_input = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
#\nCC(=O)OC1=CC=CC=C1C(=O)O"
#SMILES_input = " "
smiles = st.sidebar.text_input('then press predict button', value ="CC(=O)OC1=CC=CC=C1C(=O)O")
#SMILES = SMILES.split('\n')
#smiles, msg = remove_invalid(smiles)
#st.sidebar.write(msg)
#with st.sidebar:
 #      st.button('Predict')
img=smiles_to_img(smiles)
#st.write("a logo and text next to eachother")
col1, mid, col2 = st.columns([15,0.5,15])
with col1:
    st.image(img, use_column_width=False)
with col2:
    blk=makeblock(smiles)
    render_mol(blk)
    #st.image(render_mol(blk), use_column_width=False)
#blk=makeblock(smiles)
#render_mol(blk)	
if st.sidebar.button('Predict'):
    #st.header("Structure of the smiles")
    #s=blk=makeblock(smiles)	
    generated_descriptors1= predictSingle(smiles)
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
    arr = np.zeros((0,), dtype=np.int8)
    arr1=DataStructs.ConvertToNumpyArray(fp,arr)
    arr2 =pd.DataFrame(arr)
    array = arr2.T
    #print(array.shape)
#print(fingerprints_array1)
#mols = [Chem.rdmolfiles.MolFromSmiles(SMILES_string) for SMILES_string in smiles]
    trained_model= xgb.Booster()
    trained_model.load_model('models/model_xgb_95 2.bin')
    df3=pd.concat([array,generated_descriptors1],axis=1)
    df3 = xgb.DMatrix(df3)
	
#print(df3)
    pred_rf1 = trained_model.predict(df3)
    pred_rf1 =  (pred_rf1-0.30)
    pred_rf2 =  np.round(pred_rf1,2)	
    mol_liter1   =  10**pred_rf1
    mol_liter2   = np.round(mol_liter1,2)
    #smiles1='smiles'	
    c_name    =smiles_iupac(smiles)
#mol = Chem.MolFromSmiles(SMILES)
#MolWt = Chem.Descriptors.MolWt(mol)
 
#mol_list = []
#for smiles in SMILES:
 #   mol = Chem.MolFromSmiles(smiles)
  #  mol_list.append(mol)
#MolWt = Chem.Descriptors.MolWt(mol_list)
    MolWt1     = generated_descriptors1["MolWt"]
#print(MolWt1)
    Gram_liter1  =(10**pred_rf1)*MolWt1
    #Gram_liter1 = round(Gram_liter1,2) 	
    P_sol1 =smiles_to_sol(smiles) ## calling function to get the solubility from <pubchem
#df_results = pd.DataFrame(df_results1)
    #render_mol(blk)
    data = dict(IUPAC_Name=c_name,SMILES=smiles, Predicted_LogS=pred_rf2, 
    Mol_Liter=mol_liter2,Gram_Liter=Gram_liter1,Experiment_Solubility_PubChem=P_sol1)
    df = pd.DataFrame(data, index=[0])
    #df.round(decimals = 3)
    #st.table(df)

    # Custom formatting



    #df.round(4)
    st.header('Predicted LogS values for single smiles')
    st.table(df.style.format({"Predicted_LogS": "{:.2f}","Mol_Liter":"{:.2f}","Gram_Liter":"{:.2f}"}))
    #df
    #st.write('Good Morning') #displayed when the button is clicked
    st.header('Computed molecular descriptors')
    generated_descriptors1 # Skips the dummy first item

else:
    st.write('Note for users - 1>Enter Single smiles and click on predict button') #displayed when the button is unclicked



#smiles = st.text_input("label goes here",value="",autocomplete=None )

#SMILES = SMILES.split('\n')
#smiles, msg = remove_invalid(smiles)
#st.sidebar.write(msg)
#smiles = list(filter(None, SMILES))

#df_results1 = pd.DataFrame(smiles, columns=['smiles'])
#df_results1["SMILES"] = smiles
#df_results1["Predicted - LogS"]=pred_rf1
#df_results1=df_results1.round(3)
#df_results1["Mol/Liter"]=mol_liter1
#df_results1["Gram/Liter"]=Gram_liter1
#df_results1["Experiment Solubility-PubChem"]=P_sol1

st.sidebar.write("""---------**OR**---------""")
st.sidebar.write("""**Upload a 'csv' file with a column named 'SMILES'** (Max:2000)""")
uploaded_file = st.sidebar.file_uploader("Choose a file")
if st.sidebar.button('Prediction for input file'):
    #uploaded_file = st.sidebar.file_uploader("Choose a file")
    #data = pd.read_csv(uploaded_file)
    #SMILES=data["SMILES"]   

#uploaded_file = st.sidebar.file_uploader("Choose a file")
#if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # data
    SMILES=data["SMILES"]
    generated_descriptors = generate(SMILES)
    #rf_model_import = pickle.load(open('models/model_rf_93.pkl', 'rb'))
    mols = [Chem.rdmolfiles.MolFromSmiles(SMILES_string) for SMILES_string in SMILES]
#Convert training molecules into training fingerprints
    bi = {}
    fingerprints = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, bitInfo= bi, nBits=512) for m in mols]

#Convert training fingerprints into binary, and put all training binaries into arrays
    import numpy as np 
    import pubchempy as pcp

    fingerprints_array = []
    for fingerprint in fingerprints:
        array = np.zeros((1,), dtype= int)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        fingerprints_array.append(array)

    fingerprints_array=pd.DataFrame(fingerprints_array)
    df1=pd.concat([fingerprints_array,generated_descriptors],axis=1)
    #pred_rf = trained_model.predict(df1)
    #mol_liter =10**pred_rf

#print(df1)
### Funtion to get data from Pubchem 

    trained_model= xgb.Booster()
    trained_model.load_model('models/model_xgb_95 2.bin')
#predict test data (MLP,XGB,RF)
#pred_mlp = mlp_model_import.predict(df1)   
    df3 = xgb.DMatrix(df1)
#pred_xgb = trained_model.predict(df3)
    pred_rf = trained_model.predict(df3)
    pred_rf= pred_rf-0.30	
    #pred_rf1=np.around(pred_rf, decimals=2)	
    mol_liter =10**pred_rf
    #mol_liter1 =np.around(mol_liter, decimals=2)	
#mol = Chem.MolFromSmiles(SMILES)
#MolWt = Chem.Descriptors.MolWt(mol)
 
#mol_list = []
#for smiles in SMILES:
 #   mol = Chem.MolFromSmiles(smiles)
  #  mol_list.append(mol)
#MolWt = Chem.Descriptors.MolWt(mol_list)
    MolWt = generated_descriptors["MolWt"]
    Gram_liter=(10**pred_rf)*MolWt 
    P_sol=[] ## List where I am storing the solubility data 
    for i in range(len(SMILES)):
        try:
    #time.sleep(1) # Sleep for 3 seconds
          sol=smiles_to_sol(SMILES[i]) ### Function to get the data from PubChem 
          P_sol.append(sol)
        except AttributeError as e:
         sol=='No string'
         P_sol.append(sol)
#exp_sol = P_sol
#calculate consensus
#pred_consensus=(pred_mlp+pred_xgb+pred_rf)/3
# predefined_models.get_errors(test_logS_list,pred_enseble)

# results=np.column_stack([pred_mlp,pred_xgb,pred_rf,pred_consensus])
    df_results = pd.DataFrame(SMILES, columns=['SMILES'])
    df_results["Predicted - LogS"]=pred_rf
    df_results=df_results.round(3)
    df_results["Mol/Liter"]=mol_liter
    df_results["Gram/Liter"]=Gram_liter
    df_results["Experiment Solubility-PubChem"]=P_sol

    #df
# df_results.to_csv("results/predicted-"+test_data_name+".csv",index=False)
#st.header('Predicted LogS values for single smiles')
#df
    st.header('Predicted LogS values for a file')
#df_results # Skips the dummy first item
    df_results

    csv = df_results.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings
    linko= f'<a href="data:file/csv;base64,{b64}" download="aqsolpred_predictions.csv">Download csv file</a>'
    st.markdown(linko, unsafe_allow_html=True)
 
    st.header('Computed molecular descriptors')
    generated_descriptors # Skips the dummy first item  
else:
    st.write('2>Click on browse files and enter csv files with more than one smiles and then click on predict with input files button')
     #SMILES = ['CN1C=NC2=C1C(=O)N(C(=O)N2C)C','CC(=O)OC1=CC=CC=C1C(=O)O']   
     #SMILES=pd.DataFrame(SMILES)
     #SMILES = SMILES.iloc[:,0]
# st.header('Input SMILES')
# SMILES[1:] # Skips the dummy first item

# Use only top 300
#if len(SMILES)>2000:
 #   SMILES=SMILES[0:2000]
	
## Calculate molecular descriptors
#generated_descriptors = generate(SMILES)

#Import pretrained models
#mlp_model_import = pickle.load(open('models/model_reg_94.pkl', 'rb'))
#trained_model= xgb.Booster()
#trained_model.load_model('models/model_xgb_95 2.bin')
#rf_model_import = pickle.load(open('models/model_rf_93.pkl', 'rb'))
#mols = [Chem.rdmolfiles.MolFromSmiles(SMILES_string) for SMILES_string in SMILES]
#Convert training molecules into training fingerprints
#bi = {}
#fingerprints = [Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, bitInfo= bi, nBits=512) for m in mols]

#Convert training fingerprints into binary, and put all training binaries into arrays
#import numpy as np 
#import pubchempy as pcp

#fingerprints_array = []
#for fingerprint in fingerprints:
#        array = np.zeros((1,), dtype= int)
 #       DataStructs.ConvertToNumpyArray(fingerprint, array)
  #      fingerprints_array.append(array)

#fingerprints_array=pd.DataFrame(fingerprints_array)
#df1=pd.concat([fingerprints_array,generated_descriptors],axis=1)
#pred_rf = rf_model_import.predict(df1)
#mol_liter =10**pred_rf

#print(df1)
### Funtion to get data from Pubchem 


#predict test data (MLP,XGB,RF)
#pred_mlp = mlp_model_import.predict(df1)   
#df3 = xgb.DMatrix(df1)
#pred_xgb = trained_model.predict(df3)
#pred_rf = rf_model_import.predict(df1)
#mol_liter =10**pred_rf
#mol = Chem.MolFromSmiles(SMILES)
#MolWt = Chem.Descriptors.MolWt(mol)
 
#mol_list = []
#for smiles in SMILES:
 #   mol = Chem.MolFromSmiles(smiles)
  #  mol_list.append(mol)
#MolWt = Chem.Descriptors.MolWt(mol_list)
#MolWt = generated_descriptors["MolWt"]
#Gram_liter=(10**pred_rf)*MolWt 
#P_sol=[] ## List where I am storing the solubility data 
#for i in range(len(SMILES)): ## Dataframe which has SMILES 
 # try:
    #time.sleep(1) # Sleep for 3 seconds
  #  sol=smiles_to_sol(SMILES[i]) ### Function to get the data from PubChem 
   # P_sol.append(sol)
  #except AttributeError as e:
   # sol=='No string'
    #P_sol.append(sol)
#exp_sol = P_sol
#calculate consensus
#pred_consensus=(pred_mlp+pred_xgb+pred_rf)/3
# predefined_models.get_errors(test_logS_list,pred_enseble)

# results=np.column_stack([pred_mlp,pred_xgb,pred_rf,pred_consensus])
#df_results = pd.DataFrame(SMILES, columns=['SMILES'])
#df_results["Predicted - LogS"]=pred_rf
#df_results=df_results.round(3)
#df_results["Mol/Liter"]=mol_liter
#df_results["Gram/Liter"]=Gram_liter
#df_results["Experiment Solubility-PubChem"]=P_sol

    #df
# df_results.to_csv("results/predicted-"+test_data_name+".csv",index=False)
#st.header('Predicted LogS values for single smiles')
#df
#st.header('Predicted LogS values')
#df_results # Skips the dummy first item
#df_results
# download=st.button('Download Results File')
# if download:
#csv = df_results.to_csv(index=False)
#b64 = base64.b64encode(csv.encode()).decode()  # some strings
#linko= f'<a href="data:file/csv;base64,{b64}" download="aqsolpred_predictions.csv">Download csv file</a>'
#st.markdown(linko, unsafe_allow_html=True)
 
#st.header('Computed molecular descriptors')
#generated_descriptors # Skips the dummy first item


st.write("""
# About AqSolPred1
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

** For any feedback or suggestion please write me -- mushtaq.ali@kit.edu                                                                                         
**Contact over Linkdin :** [Mushtaq Ali](www.linkedin.com/in/mushtaq-ali/)
""")



