import torch
import numpy as np
import shap
import io
import matplotlib.pyplot as plt
import pandas as pd
# from torch.utils.data import DataLoader
# from parser import get_dataset, IrDataset



def convert_to_binary(predictions, threshold):
    main_list = []
    for i in predictions:
        results = (i>threshold).float()*1
        main_list.append(results)
    return main_list


def shapit(model, spectra_obj, predicted_groups, actual_groups, class_list, device):

    dataset = pd.read_pickle('infrared/model_data/testdf3.pkl')
    reshaped = np.vstack([j.reshape(1,1,-1) for j in dataset['spectrum'].sample(n=100, replace=False).values])
    print(reshaped.shape)
    e = shap.DeepExplainer(model,[torch.tensor(reshaped[:,:,1200:]).float().to(device),torch.tensor(reshaped[:,:,:1200]).float().to(device)])

    addarr = np.zeros(600)

    stack17 = []
    shap_values = e.shap_values([torch.tensor(spectra_obj[0]).float().to(device),torch.tensor(spectra_obj[1]).float().to(device)],check_additivity=False)


    for i in range(len(shap_values)):
        arr1 = shap_values[i][1]
        arr2 = shap_values[i][0]
        conc = np.concatenate((arr1, arr2), axis=2)
        stack17.append(conc)

    conc_spectra = np.concatenate((addarr,np.concatenate((spectra_obj[1].T, spectra_obj[0].T), axis=0).T[0,0,:]), axis = 0)
 
    xs = [k for k in range(4000)]
    yy = conc_spectra

    # Create a grid of subplots
    fig, axs = plt.subplots(6, 3, figsize=(30, 30))

    # Flatten the axs array to simplify indexing
    axs = axs.flatten()

    # Iterate over each subplot and plot the data
    for j in range(17):
        ax = axs[j]
        for i in range(len(xs)-1):
            st = stack17[j][0][0]*1000
            s = np.concatenate([addarr,st], axis = 0)

            s[(0 < s) & (s < 1)] = np.nan
            s[(0 > s) & (s > -1)] = np.nan
            
            if s[i] > 0:
                alpha = s[i]
                if alpha > 1 :
                    alpha = 1
                ax.fill_between([xs[i], xs[i+1]], 0, [yy[i], yy[i+1]], facecolor='red', interpolate=False, alpha=alpha)
            elif s[i] < 0:
                alpha = -s[i]
                if alpha > 1 :
                    alpha = 1
                ax.fill_between([xs[i], xs[i+1]], 0, [yy[i], yy[i+1]], facecolor='blue', interpolate=False, alpha=alpha)
        ax.set_title(class_list[j],fontsize=18)
        
    
        ax.plot(yy, color='black', alpha=0.2)

        print(yy.shape)
        ax.set_xlim(600, 4000)

    for j in range(17, len(axs)):
        axs[j].axis('off')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('infrared/images/tmp.png')
    # plt.title(batch['smi'][samplenum], fontsize=20)
    # plt.tight_layout()
    # plt.show()

    return 1



def infer(spectra_obj, labels):

    class_list = ['alkane', 'methyl', 'alkene',
       'alkyne', 'alcohols', 'amines', 'nitriles', 'aromatics',
       'alkyl halides', 'esters', 'ketones', 'aldehydes', 'carboxylic acids',
       'ether', 'acyl halides', 'amides', 'nitro']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('infrared/model_data/split_1200_bnorm_mix_sub_3.pt',map_location=device)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(spectra_obj[0]).float().to(device), torch.tensor(spectra_obj[1]).float().to(device))
        binary_preds = convert_to_binary(logits.cpu(), torch.tensor([0.5]))
        int_list = [int(x) for x in binary_preds[0].tolist()]
        predicted_groups = [x for x, y in zip(class_list, int_list) if y == 1]
        actual_groups =  [x for x, y in zip(class_list, labels) if y == 1]

    shs = shapit(model, spectra_obj, predicted_groups, actual_groups, class_list, device)
    return [predicted_groups , actual_groups]
