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












def calculate_descriptors(smiles):
    """
    Calculates the descriptors for a given molecule.
    """
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    generated_features = []
    for sm in smiles:
        mol = MolFromSmiles(sm)
        if mol:
            generated_features.append(calc.CalcDescriptors(mol))
        else:
            generated_features.append(None)
    return pd.DataFrame(generated_features, columns=calc.GetDescriptorNames(), index=smiles)


def load_original_dataset():
    """
    Loads the original dataset.
    """
    df = pd.read_csv('data/processed_logs.csv', header=0, sep=',')
    X = df.drop(['smiles', 'logS'], axis=1)
    Y = df['logS']
    return X, Y


def remove_invalid(smiles):
    """
    Removes invalid molecules from the dataset.
    """
    valid = [sm for sm in smiles if MolFromSmiles(sm)]
    if len(valid) == len(smiles):
        return smiles, "All provided SMILES are valid!"
    return valid, "Some SMILES are invalid! Showing results for valid SMILES only!"


st.write("""
# MOLECULAR SOLUBILITY PREDICTION WEB APP
This app predicts the **Solubility (LogS)** values of molecules!
***
""")

######################
# SIDE PANEL
######################

st.sidebar.header('USER INPUTS:')

# Read SMILES input
SMILES_input = "CC(=O)OC1=CC=CC=C1C(=O)O\n C1=CC=C(C=C1)C=O"
#SMILES_input = ""
SMILES = st.sidebar.text_area("SMILES input:", SMILES_input)
SMILES = SMILES.split('\n')
SMILES, msg = remove_invalid(SMILES)
st.sidebar.write(msg)

######################
# MAIN PANEL
######################

st.header('Input SMILES')
SMILES

model = st.sidebar.radio(
    "Which model do you want to use?",
    ('Random Forest', 'XGBoost'))


# Calculate molecular descriptors
st.header('Computed molecular descriptors')
X = generate(SMILES)
X
#print(X)
#orig_X, orig_Y = load_original_dataset()
#Turning SMILES into Explicit Bit Vectors (RDKit prefered format)
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
df1=pd.concat([fingerprints_array,X],axis=1)
print(df1)

if model == 'Random Forest':
    ######################
    # Pre-built model
    ######################

    # Reads saved model
    trained_model = pickle.load(open('models/model_rf_93.pkl', 'rb'))

    # Apply model to make predictions
    #SMILES = st.sidebar.text_area("SMILES input:", SMILES_input)
    df1 = df1.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

    predictions = trained_model.predict(df1)
    Mol_liter = 10**(predictions)

    st.header('Predicted LogS values')
    import pandas as pd
    df = pd.DataFrame({
    'smiles':[SMILES],   
     'Predicted LogS': [predictions],
     'Mol_Liter': [Mol_liter]
    })

    df

    #preds = pd.DataFrame(predictions, columns=['Predicted LogS'], index=SMILES)
    #preds

    #preds= pd.DataFrame({'SMILES':SMILES,'Predicted LogS' : predictions, 'Mol_Liter': Mol_liter}, index=[0])
    #preds = pd.DataFrame(predictions,Mol_liter, columns=['Predicted LogS','Mol/Liter'], 
    #index=SMILES)
    #preds


    
elif model == 'XGBoost':
    ######################
    # Pre-built model
    ######################

    # Reads saved model

    trained_model= xgb.Booster()
    trained_model.load_model('models/model_xgb_95.bin')

    #trained_model = xgb.XGBRegressor()
    #trained_model.load_model("models/model_xgb_95.json")

    #trained_model = pickle.load(open('models/model_xgb_95.pkl', 'rb'))
    #loaded_model_xgb = pickle.load(open("/content/drive/MyDrive/model_xgb_95.pkl", 'rb'))


    # Apply model to make predictions
    #SMILES = st.sidebar.text_area("SMILES input:", SMILES_input)
    df1 = df1.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    df1 = xgb.DMatrix(df1)

    predictions = trained_model.predict(df1)
    Mol_liter = 10**(predictions)

    st.header('Predicted LogS values')
    import pandas as pd
    df2 = pd.DataFrame({
     'smiles':[SMILES],
     'Predicted LogS': [predictions],
     'Mol_Liter': [Mol_liter]
    })
    df2
    
st.sidebar.write("""---------**OR**---------""")
st.sidebar.write("""**Upload a file with a column named 'SMILES'** (Max:2000)""")

   
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # data
    SMILES=data["SMILES"]  
    
    #preds= pd.DataFrame({'SMILES':SMILES,'Predicted LogS' : predictions, 'Mol_Liter': mol_liter}, index=[0])
    #preds = pd.DataFrame(predictions, columns=['Predicted LogS'], index=SMILES)
    #preds
if len(SMILES)>2000:
    SMILES=SMILES[0:2000]
	
## Calculate molecular descriptors
generated_descriptors = generate(SMILES)

#Import pretrained models
#mlp_model_import = pickle.load(open('aqsolpred_mlp_model.pkl', 'rb'))
#xgboost_model_import = pickle.load(open('aqsolpred_xgb_model.pkl', 'rb'))
rf_model_import = pickle.load(open('models/model_rf_93.pkl', 'rb'))
mols = [Chem.rdmolfiles.MolFromSmiles(SMILES_string) for SMILES_string in SMILES]
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
data=pd.concat([fingerprints_array,generated_descriptors],axis=1)
#print(data)


#predict test data (MLP,XGB,RF)
#pred_mlp = mlp_model_import.predict(generated_descriptors)   
#pred_xgb = xgboost_model_import.predict(generated_descriptors)
pred_rf = rf_model_import.predict(data)   
#calculate consensus
#pred_consensus=(pred_mlp+pred_xgb+pred_rf)/3
# predefined_models.get_errors(test_logS_list,pred_enseble)

# results=np.column_stack([pred_mlp,pred_xgb,pred_rf,pred_consensus])
df_results = pd.DataFrame(SMILES, columns=['SMILES'])
df_results["LogS"]=pred_rf
df_results=df_results.round(3)

# df_results.to_csv("results/predicted-"+test_data_name+".csv",index=False)

st.header('Predicted LogS values')
df_results # Skips the dummy first item

# download=st.button('Download Results File')
# if download:
csv = df_results.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()  # some strings
linko= f'<a href="data:file/csv;base64,{b64}" download="solubility_predictions.csv">Download csv file</a>'
st.markdown(linko, unsafe_allow_html=True)
        