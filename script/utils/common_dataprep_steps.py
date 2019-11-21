# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:52:34 2019

@author: JARD
"""

import re
import sys, os
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

os.environ["original_path"] = os.getcwd().split("\\dataanalytics")[0]

sys.path.append(os.environ["original_path"] + "/dataanalytics/utils/script")
import emblem_integrator as ei
import emblem_utils as eu
from win32com.client import DispatchEx

def perform_pca(X, k):
    
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=k)
    pca.fit(X)
    print("explained variance with PCA {0}".format(pca.explained_variance_ratio_))  
    
#    sns.lineplot(range(len(pca.singular_values_)), pca.explained_variance_ratio_, lw=1)
#    plt.show()
    
    return pca.transform(X)


def automatic_dataprep(new_data, Y_label):
    
    """
    this function aims at doing the EDA in an automatic way based on the ouput variable
    Aim at saving time depending on cases to study
        0) Structure of the dataset
        1) Study the output variable
        2) Missing values observation
        3) Outliers detections
        4) Modalities description
        5) Time dependant variable analysis
        2) Get the most significance variables vs output
    """
    
    
    
    # =============================================================================
    #     ##### take care of the target to be sure it is real
    # =============================================================================
    new_data[[Y_label, "Insurer"]].hist(figsize=(10,10), bins = 30, alpha=0.8, sharex= True, sharey=True, grid= True)
    plt.savefig("/".join([args["path"], args["date"], "02 results", args["cover"], "images", "output_distribution.png"]))
    
    #### delete features with no variance
    to_drop = []
    for col in new_data.columns:
        if len(new_data[col].unique())==1:
            to_drop.append(col)
    print("Columns with no variance : {0}".format(to_drop))
    new_data = new_data.drop(to_drop, axis=1)
    
    # =============================================================================
    #     ###### encode variables having 2 modalities as a dummy : e.g Male/Female ----> 0/1 
    # =============================================================================
    dico_prep = pd.DataFrame([],columns=["0","1","dummyfied"])
    for col in new_data.columns: 
        if col !="Insurer":
            if len(new_data[col].unique()) == 2 and new_data[col].dtype == "O":
               counts_values = new_data[col].fillna(-1).value_counts().sort_index().index
               if pd.isnull(new_data[col].fillna(-1).unique()).sum() == 0:
                   dico = {counts_values[0]: 0, counts_values[1]: 1}
                   dico_prep.loc[col, "0"] = counts_values[0]
                   dico_prep.loc[col, "1"] = counts_values[1]
                   dico_prep.loc[col, "dummyfied"] = "-"
                   new_data[col] = new_data[col].map(dico)
               else: ### 1 nan value, one unique | keep this feature as nan can be seen as a new modality
                   dico = {counts_values[0]: 1}
                   dico_prep.loc[col, "1"] = counts_values[0]
                   new_data[col] = new_data[col].map(dico)
    print("Columns dummyfied = {0}".format(dico_prep))
       
    # =============================================================================
    #     ##### transform object into dummy variables
    # =============================================================================
    cols_object = [x for x in new_data.columns if new_data[x].dtypes == "O"]
    supps = []
    for col in cols_object:
        if col != 'sample_number':
            if len(new_data[col].unique()) < 30 and len(new_data[col].unique()) > 1:
                new_data[col]=new_data[col].str.lower()
                a = pd.get_dummies(new_data[col], prefix = col, dummy_na=False)
                new_data = pd.concat([a, new_data], axis = 1)
                dico_prep.loc[col, "dummyfied"] = new_data[col].fillna(-1).unique()
                dico_prep.loc[col, "0"] = "-"
                dico_prep.loc[col, "1"] = "-"
            else:
                supps.append(col)
            del new_data[col]
    print("Columns with too many modalities, deleted = {0}".format(supps))
    
    new_data.columns = [x.replace(" ", "_") for x in new_data.columns]
    
    return new_data, dico_prep


def init_emblem():
    try:
        EMB= DispatchEx('Emblem_Modeller4.Application')
        EMB.Application.AppActivate()
    except Exception:
        import os
        os.system('taskkill /f /im EMBLEM_Modeller4.exe')
        init_emblem()
        pass
    return EMB


def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False


def predict_glm(bid, predicted_GLM):
    
    if len(bid["Response"].unique()) == 2: ### logistic regression
        predicted = 1/(1+np.exp(-1*predicted_GLM.sum(1)))*bid['Weight']
    elif bid["Response"].max() > 100: ### gamma log regression
        predicted = np.exp(-1*predicted_GLM.sum(1))*bid['Weight']
    else: #### poisson regression 
        predicted = np.exp(-1*predicted_GLM.sum(1))*bid['Weight']

    return predicted


def get_emblem_preds(EMB, path, bid_path, emb_path):
    
    ### Import the bid and fac files
    bid =eu.parse_fac_bid(path + '/{0}.Fac'.format(bid_path), path + '/{0}.Bid'.format(bid_path))
    EMB.OpenModelFile(path +'/{0}.emb'.format(emb_path))
    betas, base = ei.RFs_betas(EMB)
    predicted_GLM=eu.predict_from_diz(bid, betas, base)
    
    ### transform integer values from the bid files and map them to their original values
    with open(path + '/{0}.Fac'.format(bid_path), "r") as f:
        file = f.read()
    
    numerical = re.split("\n[0-9]+\t[0-9]", file.split("*** Factor Rules Section ***")[0])
    
    dico_encoding = {}
    for i in range(1, len(numerical)):
        dico_encoding[numerical[i-1].split("\n")[-1]] = {}
        for j, value in enumerate(numerical[i].split("\n")[:-1]):
            if j>=1:
                dico_encoding[numerical[i-1].split("\n")[-1]][j] = float(value) if isfloat(value) else value
                
    for key, value in dico_encoding.items():
        bid[key] = bid[key].map(value)
    
    ### predict the GLM values for the bid dataset    
    bid["GLM_PREDICTION"] = predict_glm(bid, predicted_GLM)
    bid.columns = [x.upper() for x in bid.columns]
            
    return bid, predicted_GLM.columns
