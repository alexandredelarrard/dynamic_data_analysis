# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 14:00:49 2018

@author: User
"""

import pandas as pd
import glob
import numpy as np
import os
import re
import unicodedata
import tqdm
import time

def import_data_atp(path):
    
    t0 = time.time()
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
    
    print("[{0}s] 4) Import ATP dataset ".format(time.time() - t0))
            
    return data.reset_index(drop=True)


def merge_match_ID(data1, key):
    
    t0 = time.time()
    matching_IDS = pd.read_csv(os.environ["DATA_PATH"] + "/clean_datasets/players/match_originID_atpID.csv")
    datamerge = pd.merge(data1, matching_IDS, on =key, how = "left")
    
    print("new data shape is {0}".format(datamerge.shape))
    print("[{0}s] 3) Merge ATP ID vs Origin ID for origin dataset ".format(time.time() - t0))
    
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
        return np.nan
    

def merge_origin_atp(data_orig, data_atp, common_key = "ATP_ID"):
    
    t0 = time.time()
    
    total_data = pd.merge(data_orig, data_atp, on = common_key, how= "left")
    
    for col in ["Winner", "Loser" , "winner_name", "loser_name"]:
        total_data[col] = total_data[col].apply(lambda x : strip_accents(x).replace("'","").replace("-"," ").replace(".","").lstrip().rstrip().lower())
        
    total_data["win_lose_orig"] = total_data["Winner"] + " " + total_data["Loser"]
    total_data["win_lose_atp"] = total_data["winner_name"] + " " + total_data["loser_name"]
    
    total_data["bool"] = total_data[["win_lose_orig","win_lose_atp"]].apply(lambda x : len(set.intersection(set(x[0].split(" ")), set(x[1].split(" ")))), axis=1)
    print("matching results : \n {0}".format(total_data["bool"].value_counts()))
   
    total_data = total_data.rename(columns = {"Date_x" : "Date", "Surface_x": "Surface", "Tournament_x" : "Tournament"})

    ### suppress  as not in both datasets
    total_data = total_data.loc[total_data["ATP_ID"] != -1] 
    total_data["best_of"] = total_data["best_of"].astype(int)
    
    #### take care of missing values
    total_data = fill_in_missing_values(total_data)

    total_data = total_data[["ATP_ID", "ORIGIN_ID",  'winner_id', 'loser_id', "Date", "Date_start_tournament", "winner_name", "loser_name", "score", "WRank",  "LRank", "Surface", "Tournament", 'ATP', "City", "tourney_name", 'draw_size', "Court", "Comment", 'best_of', 'Round', "round", "Prize", "Currency",   #### match macro desc + tournament desc
                              #### players desc
                            "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced", "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced", "minutes", #### match micro desc 
                             'SBL', 'SBW', 'SJL', 'SJW', 'UBL', 'UBW', 'MaxL', 'MaxW', 'PSL', 'PSW','IWL', 'IWW', 'EXL', 'EXW', 'GBL', 'GBW', 'CBL', 'CBW', 'AvgL', 'AvgW', 'B&WL', 'B&WW', 'B365L', 'B365W']] #### bets odds 
     
    total_data["winner_id"] = total_data["winner_id"].astype(int) 
    total_data["loser_id"] = total_data["loser_id"].astype(int) 
    
    print("[{0}s] 5) Merge ATP with Origin dataset ".format(time.time() - t0))
           
    return total_data


def fill_in_missing_values(total_data):
    
    total_data.loc[total_data["ATP_ID"] == 23373, "score"] = "6-1 4-6 7-5"
    
    ### take care of mismatch ranks atp wrank = atp_rank then keep wrank
    total_data.loc[pd.isnull(total_data["winner_rank"]), "winner_rank"] = total_data.loc[pd.isnull(total_data["winner_rank"]), "WRank"]
    total_data.loc[pd.isnull(total_data["loser_rank"]), "loser_rank"] = total_data.loc[pd.isnull(total_data["loser_rank"]), "LRank"]
    
    total_data.loc[pd.isnull(total_data["winner_rank_points"]), "winner_rank_points"] = total_data.loc[pd.isnull(total_data["winner_rank_points"]), "WPts"]
    total_data.loc[pd.isnull(total_data["loser_rank_points"]), "loser_rank_points"] = total_data.loc[pd.isnull(total_data["loser_rank_points"]), "LPts"]
    
    missing_stats_match = pd.read_csv(os.environ["DATA_PATH"] + "/brute_info/historical/correct_missing_values/missing_match_stats.csv")
    
    for i in tqdm.tqdm(missing_stats_match["ATP_ID"]):
        for col in ["w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced", "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced", "minutes"]:
            total_data.loc[total_data["ATP_ID"] == i, col] = missing_stats_match.loc[missing_stats_match["ATP_ID"] == i, col]
         
    return total_data
 
