# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 14:00:49 2018

@author: User
"""

import pandas as pd
import glob
import numpy as np
from datetime import datetime
import os
import re
import unicodedata


def import_data_atp(path):
    
    liste_files = glob.glob(path + "/*.csv")
    
    for i, file in enumerate(liste_files):
        if i == 0:
            data = pd.read_csv(file)
        else:
            data = pd.concat([data, pd.read_csv(file, encoding = "latin1")], axis=0)
            
    data["Date"]   = pd.to_datetime(data["tourney_date"], format = "%Y%m%d")   
    del data["tourney_date"]
    
    data = data.sort_values(["Date", "tourney_name"])
    data["ATP_ID"] = range(len(data))
    
    data.loc[data["winner_name"] == "joshua goodall", "winner_name"] = "josh goodall"
    data.loc[data["loser_name"] == "joshua goodall", "loser_name"] = "josh goodall"
            
    return data.reset_index(drop=True)


def merge_match_ID(data1, key):
    
    matching_IDS = pd.read_csv(os.environ["DATA_PATH"] + "/clean_datasets/players/match_originID_atpID.csv")
    datamerge = pd.merge(data1, matching_IDS, on =key, how = "left")
    
    print("new data shape is {0}".format(datamerge.shape))
    
    return datamerge


def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD',str(s))
                  if unicodedata.category(c) != 'Mn')


def score(x, i, j):
    try:
        return re.sub("[\(\[].*?[\)\]]", "", str(x)).split(" ")[i].split("-")[j]
    except Exception:
        return "0"
    
def compare_score(x):
    try:
        return  [int(x[0]), int(x[1]), int(x[2]), int(x[3])] == [x[4], x[5], x[6], x[7]]
    except Exception:
        return False
    

def merge_origin_atp(data_orig, data_atp, common_key = "ATP_ID"):
    
    total_data = pd.merge(data_orig, data_atp, on = common_key, how= "left")
    
    for col in ["Winner", "Loser" , "winner_name", "loser_name"]:
        total_data[col] = total_data[col].apply(lambda x : strip_accents(x).replace("'","").replace("-"," ").replace(".","").lstrip().rstrip().lower())
        
    total_data["win_lose_orig"] = total_data["Winner"] + " " + total_data["Loser"]
    total_data["win_lose_atp"] = total_data["winner_name"] + " " + total_data["loser_name"]
    
    total_data["bool"] = total_data[["win_lose_orig","win_lose_atp"]].apply(lambda x : len(set.intersection(set(x[0].split(" ")), set(x[1].split(" ")))), axis=1)
    print("matching results : \n {0}".format(total_data["bool"].value_counts()))
    
    total_data["score"] = total_data["score"].fillna("0-0 0-0")
    total_data["W1_atp"] = total_data["score"].apply(lambda x : score(x, 0, 0))
    total_data["L1_atp"] = total_data["score"].apply(lambda x : score(x, 0, 1))
    total_data["W2_atp"] = total_data["score"].apply(lambda x : score(x, 1, 0))
    total_data["L2_atp"] = total_data["score"].apply(lambda x : score(x, 1, 1))
    total_data = total_data.fillna(0)
    
    total_data.loc[total_data["W1_atp"] == "W/O"] = -1
    total_data.loc[total_data["W1_atp"] == ""] = -1
    total_data.loc[total_data["L1_atp"] == "W/O"] = -1
    total_data.loc[total_data["L1_atp"] == ""] = -1
    total_data.loc[total_data["W2_atp"] == "W/O"] = -1
    total_data.loc[total_data["W2_atp"] == ""] = -1
    total_data.loc[total_data["L2_atp"] == "W/O"] = -1
    total_data.loc[total_data["L2_atp"] == ""] = -1
    total_data[["W1","W2","L1","L2"]] = total_data[["W1","W2","L1","L2"]].replace("","0").replace(" ", "0").astype(int)
    total_data[["W1_atp","W2_atp","L1_atp","L2_atp"]] = total_data[["W1","W2","L1","L2"]].replace("","0").replace(" ", "0").astype(int)
#    
#    total_data["bool2"] = total_data[["W1", "W2", "L1","L2", "W1_atp", "W2_atp","L1_atp", "L2_atp"]].apply(lambda x :compare_score(x), axis=1 )
#    print(total_data["bool2"].value_counts())
    
    total_data = total_data.rename(columns = {"Date_x" : "Date", "Surface_x": "Surface", "Tournament_x" : "Tournament"})
    total_data = total_data[["ATP_ID", "ORIGIN_ID", "Date", "Date_start_tournament", "winner_name", "loser_name", "score", "WRank",  'winner_rank', "LRank", 'loser_rank', "Surface", "Tournament", "City", "tourney_name", "Court", "Comment", 'best_of', 'round', "W1_atp", "W1", "W2_atp", "W2", "L1_atp", "L1", "L2_atp", "L2"]]
    
    return total_data

 
