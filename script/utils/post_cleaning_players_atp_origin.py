# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 21:53:50 2018

@author: User
"""

import pandas as pd
import numpy as np
from utils.match_players_atp_ddb import import_data_atp

import sys
sys.path.append(r"D:\projects\tennis betting\script\data_prep")
from exctract_data1 import import_data
from create_tournament import merge_with_tournois
import unicodedata


def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD',str(s))
                  if unicodedata.category(c) != 'Mn')


def ddb():
    
    players_id_match = pd.read_csv(r"D:\projects\tennis betting\data\brute_info\players\manual_match_bdd12.csv")
    
    path = r"D:\projects\tennis betting\data\brute_info\historical\brute_info_origin"
    data1 = import_data(path)
    path_tournament = r"D:\projects\tennis betting\data\clean_datasets\tournament\tournaments.csv"
    data_orig = merge_with_tournois(data1, path_tournament)
    data_orig = data_orig[["Winner", "Loser", "Date_start_tournament", "City"]]
    data_orig["City"] = data_orig["City"].apply(lambda x: strip_accents(x))
    
    data_orig = pd.merge(data_orig, players_id_match, left_on = ['Date_start_tournament','Winner', 'Loser', 'City'], right_on = ['Date_start_tournament','Winner', 'Loser', 'City'], how= "left")
    
    path = r"D:\projects\tennis betting\data\brute_info\historical\brute_info_atp"
    players_atp = import_data_atp(path)
    players_atp = data_atp.sort_values(["Date", "tourney_name"])
    players_atp["ID_to_match"] = range(len(players_atp))
    players_atp = players_atp[["ID_to_match","Date", "winner_name", "loser_name", "tourney_name"]]

    return data_orig, players_atp


def process_data(total_data):
    
    for col in ["Winner", "Loser" , "winner_name", "loser_name"]:
        total_data[col] = total_data[col].apply(lambda x : strip_accents(x).replace("'","").replace("-"," ").replace(".","").lstrip().rstrip().lower())
        
    total_data["win_lose_orig"] = total_data["Winner"] + " " + total_data["Loser"]
    total_data["win_lose_atp"] = total_data["winner_name"] + " " + total_data["loser_name"]
    
    total_data["bool"] = total_data[["win_lose_orig","win_lose_atp"]].apply(lambda x : len(set.intersection(set(x[0].split(" ")), set(x[1].split(" ")))), axis=1)


if __name__ == "__mane__":
    
    data_orig, data_atp =  ddb()
    
    total_data = pd.merge(data_orig, data_atp, left_on="ID_to_match", right_on = "ID_to_match", how= "left")
    total_data["ORIGIN_ID"] = range(len(total_data))
    total_data["ATP_ID"]  = total_data["ID_to_match"] 
    total_data[["ATP_ID", "ORIGIN_ID"]].to_csv(r"D:\projects\tennis betting\data\brute_info\players\match_originID_atpID.csv", index= False)
    
    
    
    