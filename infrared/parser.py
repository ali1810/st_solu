import rdkit.Chem as Chem
import pandas as pd
import numpy as np
import torch
from jcamp import JCAMP_reader
from scipy import interpolate
import json
from infrared.inference import infer

THRESHOLD_IR = 0.93
THRESHOLD_NMR = 0.02
curve_begin_idx = 600
curve_correction_factor = 0.5
split_at = 1200

from torch.utils.data import Dataset

class IrDataset(Dataset):

    ''' Data Loader class for Pytorch DataLoader
    '''

    def __init__(self, df, num_classes, transform=None):
        self.df = df
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return self.df.shape[0]
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        xs = self.df.iloc[idx]['spectrum'].reshape(1, -1)

        ys = np.array([i for i in self.df.iloc[idx]['concat_label']]).astype('float').reshape(-1)

        xs = torch.from_numpy(xs).float()
        ys = torch.from_numpy(ys).float()

        ws = np.ones(ys.shape)

        sample = { 
            'xs': xs,
            'ws': ws, 
            'ys': ys 
        }

        if self.transform:
            sample = self.transform(sample)
        return sample


def concat_label(dataset):

    ''' Concat string binary labels
    Args:
        dataset : pandas dataframe
    Returns:
        dataset : pandas dataframe with additional column 

    '''

    labels = dataset[dataset.columns[4:]]
    dataset['total_func'] = labels.sum(axis=1).astype(str)
    dataset['concat_label'] = labels.astype(int).astype(str).apply(''.join, axis=1).astype(str)
    dataset.reset_index(drop=True)

    return dataset


def get_dataset(path1, mix, **kwargs):  
            
    ''' Get dataset dataframe suited for dataloader
    Args:
        path1 : dataset pickle file path
        mix : If the dataset is mixed (Boolean)
        path2 : dataset pickle file path to mix

    Returns:
        dataset : pandas dataframe

    '''
    if mix == True :
        if len(kwargs['path2']) > 0 :
            dataset_1 = pd.read_pickle(path1)
            dataset_2 = pd.read_pickle(kwargs['path2'])
            dataset = pd.concat([dataset_1, dataset_2], ignore_index=True)
        else:
            raise FileNotFoundError('Mention path2 if mix = True')
    elif mix == False :
        dataset = pd.read_pickle(path1)
    
    dataset = concat_label(dataset)
    return dataset


def identify_fg(smiles):
    '''Identify the presence of functional groups present in molecule 
       denoted by inchi
    Args:
        root: (string) path to spectra data
        files: (list) jdx files present in root
        save_path: (string) path to store csv file
        bins: (np.array) used for standardizing
        is_mass: (bool) whether data being parsed is Mass or IR
    Returns:
        mol_func_groups: (list) contains binary values of functional groups presence
                          None if inchi to molecule conversion returns warning or error
    '''
    func_grp_smarts = {'alkane':'[CX4;H0,H1,H2,H4]','methyl':'[CH3]','alkene':'[CX3]=[CX3]','alkyne':'[CX2]#C',
                   'alcohols':'[#6][OX2H]','amines':'[NX3;H2,H1;!$(NC=O)]', 'nitriles':'[NX1]#[CX2]', 
                   'aromatics':'[$([cX3](:*):*),$([cX2+](:*):*)]','alkyl halides':'[#6][F,Cl,Br,I]', 
                   'esters':'[#6][CX3](=O)[OX2H0][#6]', 'ketones':'[#6][CX3](=O)[#6]','aldehydes':'[CX3H1](=O)[#6]', 
                   'carboxylic acids':'[CX3](=O)[OX2H1]', 'ether': '[OD2]([#6])[#6]','acyl halides':'[CX3](=[OX1])[F,Cl,Br,I]',
                   'amides':'[NX3][CX3](=[OX1])[#6]','nitro':'[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]'}

    func_grp_structs = {func_name : Chem.MolFromSmarts(func_smarts) for func_name, func_smarts in func_grp_smarts.items()}

    try:
        #Convert inchi to molecule
        mol = Chem.MolFromSmiles(smiles)#, treatWarningAsError=True)

        mol_func_grps = []

        #populate the list with binary values
        for _, func_struct in func_grp_structs.items():
            struct_matches = mol.GetSubstructMatches(func_struct)
            contains_func_grp = int(len(struct_matches)>0)
            mol_func_grps.append(contains_func_grp)
        return mol_func_grps
    except:
        return None

def order(x_i, y_i):
    x_o = x_i
    y_o = y_i

    first_x = x_i[0]
    last_x = x_i[-1]
    if first_x > last_x:
        x_o = x_o[::-1]
        y_o = y_o[::-1]
    return x_o, y_o


def concat_boundary(x_i, y_i):
    x_head = np.array([0.0])
    x_tail = np.array([4000.0])
    x_o = np.concatenate([x_head, x_i, x_tail])

    y_head = np.array([y_i[0]])
    y_tail = np.array([y_i[-1]])
    y_o = np.concatenate([y_head, y_i, y_tail])
    return x_o, y_o


def sampling(x_i, y_i):
    f = interpolate.interp1d(x_i, y_i)
    x_o = np.linspace(0, 3999, 4000, endpoint=True)
    y_o = f(x_o)
    return x_o[curve_begin_idx:], y_o[curve_begin_idx:]


def normalize(x_i, y_i):
    max_y = np.max(y_i)
    min_y = np.min(y_i)
    height = max_y - min_y

    x_o = x_i
    y_o = (y_i - min_y) / height
    return x_o, y_o


def to_absorption(x_i, y_i):
    x_o = x_i
    y_o = -np.log10(y_i)
    
    return x_o, y_o


def chop(x_i, y_i):
    x_o = x_i[0:]
    y_o = y_i[0:]
    
    return x_o, y_o

def parse_jdx(filepath, is_abs):

    data_dict = JCAMP_reader(filepath)
    x = data_dict['x']
    y = data_dict['y']

    #order
    x_orig, y_orig = order(x, y)

    # concatenate
    x_basis, y_basis = concat_boundary(x_orig, y_orig)

    #interpolate
    x_new, y_new = sampling(x_basis, y_basis)
    y_new = y_new ** curve_correction_factor

    if is_abs == 'absorption' : 
        x_new_norm_abs, y_new_norm_abs = x_new, y_new
    else :
        x_new_norm_abs, y_new_norm_abs = to_absorption(x_new, y_new)


    #normalized
    x_new_norm, y_new_norm = normalize(x_new_norm_abs, y_new_norm_abs)

    # chop
    x_nnac, spectra = chop(x_new_norm, y_new_norm)
    
    return spectra
    
def preprocess_spectra(spectra, split_at):
    spectra_obj = [spectra[split_at:].reshape(1,1,-1), spectra[:split_at].reshape(1,1,-1)]
    return spectra_obj

def process_input(filepath, smiles, split_at, is_abs):

    spectra = parse_jdx(filepath, is_abs)
    spectra_obj = preprocess_spectra(spectra, split_at)
    labels = identify_fg(smiles)
    output_preds = infer(spectra_obj, labels)
    
    print(is_abs)
    # json_str = json.dumps(data_list)
    
    return output_preds
