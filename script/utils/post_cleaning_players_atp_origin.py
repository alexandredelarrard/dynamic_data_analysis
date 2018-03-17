# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 21:53:50 2018

@author: User
"""

import pandas as pd
import numpy as np
import glob

import sys
sys.path.append(r"C:\Users\JARD\Documents\Projects\tennis\v09032018\scripts")
from data_prep_historical_data import import_data
from data_prep_tournament_data import merge_with_tournois
import unicodedata
import re
import tqdm

def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD',str(s))
                  if unicodedata.category(c) != 'Mn')



def import_data_atp(path):
    
    liste_files = glob.glob(path + "/*.csv")
    
    for i, file in enumerate(liste_files):
        if i == 0:
            data = pd.read_csv(file)
        else:
            data = pd.concat([data, pd.read_csv(file, encoding = "latin1")], axis=0)
            
    data["Date"]   = pd.to_datetime(data["tourney_date"], format = "%Y%m%d")   
    del data["tourney_date"]
    
    data.loc[data["winner_name"] == "joshua goodall", "winner_name"] = "josh goodall"
    data.loc[data["loser_name"] == "joshua goodall", "loser_name"] = "josh goodall"
            
    return data.reset_index(drop=True)


def ddb():
    
    players_id_match = pd.read_csv(r"C:\Users\JARD\Documents\Projects\tennis\v09032018\data\manual_match_bdd12.csv")
    players_id_match["Date_start_tournament"] =  players_id_match["Date_start_tournament"].astype(str)
    
    path = r"C:\Users\JARD\Documents\Projects\tennis\v09032018\data\historical"
    data1 = import_data(path)
    path_tournament = r"C:\Users\JARD\Documents\Projects\tennis\v09032018\data\tournament\tournaments.csv"
    data_orig = merge_with_tournois(data1, path_tournament)
    data_orig["ID"] = range(len(data_orig))
    
    data_orig["Date_start_tournament"] = data_orig["Date_start_tournament"].astype(str)
    data_orig["City"] = data_orig["City"].apply(lambda x: strip_accents(x))
    data_orig = pd.merge(data_orig, players_id_match, left_on = ['Date_start_tournament','Winner', 'Loser', 'City'], right_on = ['Date_start_tournament','Winner', 'Loser', 'City'], how= "left")
    data_orig = data_orig.drop_duplicates("ID")
    data_orig = data_orig[["ID", "ID_to_match", "Date_start_tournament", "Winner", "Loser", "W1", "W2", "W3", "L1", "L2", "L3", "City"]]
    
    path = r"C:\Users\JARD\Documents\Projects\tennis\v09032018\data\historical"
    players_atp = import_data_atp(path)
    players_atp = players_atp.sort_values(["Date", "tourney_name"])
    players_atp["ID_to_match"] = range(len(players_atp))
    players_atp = players_atp[["ID_to_match","Date", "winner_name", "loser_name", "tourney_name", 'score']]

    return data_orig, players_atp

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
    
    
def process_data(data_orig, data_atp):
    
    total_data = pd.merge(data_orig, data_atp, left_on="ID_to_match", right_on = "ID_to_match", how= "left")
    
    for col in ["Winner", "Loser" , "winner_name", "loser_name"]:
        total_data[col] = total_data[col].apply(lambda x : strip_accents(x).replace("'","").replace("-"," ").replace(".","").lstrip().rstrip().lower())
        
    total_data["win_lose_orig"] = total_data["Winner"] + " " + total_data["Loser"]
    total_data["win_lose_atp"] = total_data["winner_name"] + " " + total_data["loser_name"]
    
    total_data["bool"] = total_data[["win_lose_orig","win_lose_atp"]].apply(lambda x : len(set.intersection(set(x[0].split(" ")), set(x[1].split(" ")))), axis=1)
    print(total_data["bool"].value_counts())
    
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
    
    total_data["bool2"] = total_data[["W1", "W2", "L1","L2", "W1_atp", "W2_atp","L1_atp", "L2_atp"]].apply(lambda x :compare_score(x), axis=1 )
    print(total_data["bool2"].value_counts())
    
    return total_data


def corrections(total_data):
    
    return total_data

if __name__ == "__mane__":
    
    data_orig, data_atp =  ddb()
    total_data = process_data(data_orig, data_atp)
    total_data.to_csv(r"C:\Users\JARD\Documents\Projects\tennis\v09032018\data\\post_cleaning_players.csv", index= False)
    
    total_data["ORIGIN_ID"] = total_data["ID"] 
    total_data["ATP_ID"]  = total_data["ID_to_match"].astype(int)
    total_data.loc[(total_data["ORIGIN_ID"] == -1) & (total_data["ATP_ID"] == -1), "ORIGIN_ID"] = total_data.loc[(total_data["ORIGIN_ID"] == -1) & (total_data["ATP_ID"] == -1), "ORIGIN_ID"].index
#    total_data[["ATP_ID", "ORIGIN_ID"]].to_csv(r"C:\Users\JARD\Documents\Projects\tennis\v09032018\data\match_originID_atpID.csv", index= False)
    
    
    name_dictionnary = {}
    for i in tqdm.tqdm(range(len(total_data))):
        name_orig = total_data.iloc[i]["Winner"]
        if name_orig not in name_dictionnary.keys():
            name_dictionnary[name_orig] = []
            
        if total_data.iloc[i]["winner_name"] != 'nan':
            name_dictionnary[name_orig] += [total_data.iloc[i]["winner_name"]]
            name_dictionnary[name_orig] = list(set(name_dictionnary[name_orig]))
            
        name_orig = total_data.iloc[i]["Loser"]
        if name_orig not in name_dictionnary.keys():
            name_dictionnary[name_orig] = []
            
        if total_data.iloc[i]["loser_name"] != 'nan':
            name_dictionnary[name_orig] += [total_data.iloc[i]["loser_name"]]
            name_dictionnary[name_orig] = list(set(name_dictionnary[name_orig]))
        
    for key in name_dictionnary.keys():
        if len(name_dictionnary[key])>1:
            print(key,name_dictionnary[key])
     
    total_data.loc[(total_data["Loser"] == "mathieu ph")&(total_data["loser_name"] == "julien mathieu")][["Date_start_tournament", "Winner", "Loser", "winner_name", "loser_name","City"]]
            
#    match = total_data[["ATP_ID", "ORIGIN_ID"]]
#    
#    path = r"C:\Users\JARD\Documents\Projects\tennis\v09032018\data\historical"
#    data1 = import_data(path)
#    data1["ORIGIN_ID"] = range(len(data1))
#    data1 = pd.merge(data1, match, on = "ORIGIN_ID")
#    data1["ID_to_match"] = data1["ATP_ID"]
#
#    process_data(data1, data_atp)