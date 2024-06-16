import streamlit as st
import os
from PIL import Image
from stmol import showmol
import py3Dmol
from rdkit import Chem
import pubchempy as pcp
from rdkit.Chem import AllChem
from urllib.request import urlopen
import sys
sys.path.append('infrared/parser.py')  
from infrared.parser import process_input


def calculate_aromatic_proportion(smiles):
    # Parse SMILES string and generate molecular representation
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Invalid SMILES
    
    # Identify aromatic atoms
    aromatic_atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms() if atom.GetIsAromatic()]
    
    # Calculate aromatic proportion
    total_atoms = mol.GetNumAtoms()
    aromatic_proportion = len(aromatic_atoms) / total_atoms
    
    return aromatic_proportion

st.header("Functional group prediction from IR spectrum")

st.markdown("<li><span style='font-size:25px'>Input molecule SMILES</span></li>", unsafe_allow_html=True)
smiles = st.text_input("")

if smiles : 
    aromatic_proportion = calculate_aromatic_proportion(smiles)

    if aromatic_proportion is not None:
        st.write("Given smiles is Valid")
    else:
        st.write("Invalid smiles")
    prop=pcp.get_properties([ 'MolecularWeight'], smiles, 'smiles')
    x = list(map(lambda x: x["CID"], prop)) 
    y=x[0]
    x = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/%s/PNG?image_size=300x300"
    url=(x % y)
    #print(url)
    img = Image.open(urlopen(url))
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    mblock = Chem.MolToMolBlock(mol)
    xyzview = py3Dmol.view(width=300,height=300)
    xyzview.addModel(mblock,'mol')
    xyzview.setStyle({'model': -1}, {"cartoon": {'color': 'spectrum'}})
    style = 'stick'
    spin = st.checkbox('Animation', value = True) 
    xyzview.spin(True)
    if spin:
        xyzview.spin(True)
    else:
        xyzview.spin(False)
        #xyzview.setStyle({'sphere':{}})
    xyzview.setBackgroundColor('#EAE5E5')
    xyzview.zoomTo()
    xyzview.setStyle({style:{'color':'spectrum'}})
    col1, mid, col2 = st.columns([15,2.5,15])
    with col1:
            st.image(img, use_column_width=False)
    with col2:
            showmol(xyzview,height=300,width=400) 


st.markdown("<li><span style='font-size:25px'>Input IR spectrum file (JCAMP)</span></li>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jdx", "dx"])

st.markdown("<li><span style='font-size:25px'>Data Format</span></li>", unsafe_allow_html=True)
is_abs = st.selectbox("", ["absorption", "transmission"])

# output_text = st.empty()

if uploaded_file is not None and smiles != "":
    button = st.button("Predict")

    if button:
        if uploaded_file:
            # Save the uploaded file to a temporary location
            file_path = os.path.join("/tmp", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process the input using the file path
            result = process_input(file_path, smiles, 1200, is_abs)
            result_list = list(result)
            st.write('Actual groups from SMILES : {}'.format(', '.join(result_list[1])))
            st.write('Predicted groups : {}'.format(', '.join(result_list[0])))
            st.markdown("<div style='text-align: center; padding-top: 30px; font-size: 20px;'>Shap analysis</div>", unsafe_allow_html=True)

# Increase image size and center it
            st.image('infrared/images/tmp.png', width=1500, use_column_width='always')

            # st.image('infrared/images/tmp.png',width=1500)
