# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 09:30:19 2018

@author: JARD
"""

import sys
sys.path.append(r"D:\projects\tennis betting\script\data_prep")
from exctract_data1 import import_data
from create_tournament import merge_with_tournois

from multiprocessing import Pool
from functools import partial
import glob
import pandas as pd
from datetime import timedelta
import unicodedata
import numpy as np
from tqdm import tqdm
import time 

tqdm.pandas(tqdm())

def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')


def parallelize_dataframe(df, function, dictionnary, njobs):
    
    df_split = np.array_split(df, njobs)
    pool = Pool(njobs)
    func = partial(function, dictionnary)
    df2 = pd.concat(pool.map(func, df_split))
    
    pool.close()
    pool.join()
    
    return df2


def import_data_atp(path):
    
    liste_files = glob.glob(path + "/*.csv")
    
    for i, file in enumerate(liste_files):
        if i == 0:
            data = pd.read_csv(file)
        else:
            data = pd.concat([data, pd.read_csv(file, encoding = "latin1")], axis=0)
            
    data["Date"]   = pd.to_datetime(data["tourney_date"], format = "%Y%m%d")   
    del data["tourney_date"]
            
    return data.reset_index(drop=True)


def data_match(data_atp, data_orig):
    data_orig["ID_to_match"] = data_orig["key"].progress_apply(lambda x : match_proba(x, data_atp))
    return data_orig


def match_proba(x, data):
    
    data_atp = data[0]
    
    sub_data = data_atp.loc[(data_atp["Date"]>= (x["Date"] - timedelta(days= 3)))&(data_atp["Date"] <= x["Date"] + timedelta(days= 3))]   
    sub_data["counts"] = sub_data["key"].apply(lambda y : sum([1 for a in x["key"] if a in y]))
    
    
    if len(sub_data["counts"])>0:
        if sum((sub_data["counts"] == max(sub_data["counts"]))*1)>1:
            return [sub_data.loc[sub_data["counts"] == max(sub_data["counts"]), "ID_to_match"].tolist()]
        elif sum((sub_data["counts"] == max(sub_data["counts"]))*1) == 1:
            return sub_data.loc[sub_data["counts"] == max(sub_data["counts"]), "ID_to_match"].values[0]
    else:
        return -1
    
    
def create_key_atp(x):
    return [x["tourney_name"].lower().replace("-"," ").replace("."," ").split(" ") +\
            x["winner_name"].lower().replace("-"," ").replace("."," ").rstrip().split(" ") + \
            x["loser_name"].lower().replace("-"," ").replace("."," ").rstrip().split(" ") + \
            [x["Date"].month] + [x["Date"].day]]
    

def create_key_orig(x):
    return [{"Date" : x["Date_start_tournament"], "key": x["City"].lower().replace("-"," ").replace("."," ").split(" ") +\
            x["Winner"].lower().replace("-"," ").replace("."," ").rstrip().split(" ") + \
            x["Winner"].lower().replace("-"," ").replace("."," ").rstrip().split(" ") + \
            x["Loser"].lower().replace("-"," ").replace("."," ").rstrip().split(" ") + \
            x["Loser"].lower().replace("-"," ").replace("."," ").rstrip().split(" ") + \
            [x["Date_start_tournament"].month] + [x["Date_start_tournament"].day]}]
            
    
if __name__ == "__main__":
    
    ### import data
    path = r"D:\projects\tennis betting\data\brute_info\historical\brute_info_origin"
    data1 = import_data(path)
    
    path = r"D:\projects\tennis betting\data\brute_info\historical\brute_info_atp"
    data_atp = import_data_atp(path)
    data_atp = data_atp.sort_values(["Date", "tourney_name"])
    
    path_tournament = r"D:\projects\tennis betting\data\clean_datasets\tournament\tournaments.csv"
    data_orig = merge_with_tournois(data1, path_tournament)
    
    ### create matching keys
    data_orig = data_orig[["Winner", "Loser", "Date_start_tournament", "City"]]
    data_orig["City"] = data_orig["City"].apply(lambda x: strip_accents(x))
    
    data_atp["ID_to_match"] = range(len(data_atp))
    data_atp = data_atp[["ID_to_match","Date", "winner_name", "loser_name", "tourney_name"]]
    data_atp["key"] = data_atp.apply(lambda x: create_key_atp(x), axis=1)["Date"]
    
    data_orig["key"] = data_orig.apply(lambda x : create_key_orig(x), axis=1)["Winner"]
    
    ### match id closest infos
    t0 = time.time()
    new_data_orig = data_match([data_atp], data_orig)
    print(time.time() - t0)
    
#    print("started parallel")
#    t0 = time.time()
#    new_data_orig = parallelize_dataframe(data_orig[:100], data_match, [data_atp], njobs=7)
#    print(time.time() - t0)
    
    new_data_orig = new_data_orig[["ID_to_match","Date_start_tournament", "Winner", "Loser", "City"]]
    new_data_orig.to_csv(r"D:\projects\tennis betting\data\brute_info\players\manual_match_bdd12_V2.csv", index= False)
    data_atp.to_csv(r"D:\projects\tennis betting\data\brute_info\players\manual_match_bdd21.csv", index= False)
        
    