# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 14:00:49 2018

@author: User
"""

import pandas as pd
import glob
import numpy as np
import os
from tqdm import tqdm
import time 

from utils.build_match_statistics_database import match_stats_main
from data_prep.extract_players import  merge_atp_players
from data_prep.missing_rank  import fill_ranks_based_origin
from data_prep.missing_stats  import fillin_missing_stats

def import_data_atp(path, redo=False):
    
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
    
    #### suppress challenger matches
    data = data.loc[data["tourney_level"] != "C"]
    
    ### suppress walkovers
    data = data.loc[~data["score"].isin(["W/O", " W/O"])]
    
    #### suppress davis cup and JO
    not_davis_index = data["tourney_name"].apply(lambda x : "Davis Cup" not in x and "Olympic" not in x)
    data = data.loc[not_davis_index]
    
    #### fill in missing scores
    data.loc[(data["tourney_id"] == "2007-533")&(pd.isnull(data["score"])), "score"] = "6-1 4-6 7-5"
    data.loc[(data["tourney_id"] == "1997-319")&(pd.isnull(data["score"])), "score"] = "6-4 6-4 6-4"
    
    data.loc[data["winner_name"] == "joshua goodall", "winner_name"] = "josh goodall"
    data.loc[data["loser_name"] == "joshua goodall", "loser_name"] = "josh goodall"
    data = data.reset_index(drop=True)
    print("[{0}s] 1) Import ATP dataset ".format(time.time() - t0))

    total_data = fill_in_missing_values(data, redo)
            
    return total_data


def fill_in_missing_values(total_data, redo):
    
    
    #### fill in missing ranks and points
    t0 = time.time()
    total_data_wrank = fill_ranks_based_origin(total_data)
    total_data_wrank = total_data_wrank.drop(["winner_seed", "winner_entry", "loser_seed", "loser_entry"],axis=1)
    print("[{0}s] 2) fill missing rank based on closest info ".format(time.time() - t0))
    
    #### add match stats on service missing
    t0 = time.time()
    total_data_wrank_stats = merge_atp_missing_stats(total_data, redo)
    print("[{0}s] 3) fill missing stats based on atp crawling matching ".format(time.time() - t0))
    
    #### fill in irreductible missingvalues based on history
    t0 = time.time()
    total_data_wrank_stats = fillin_missing_stats(total_data_wrank_stats)
    print("[{0}s] 4) fill missing stats based on previous matches ".format(time.time() - t0))
    
    #### merge with tourney ddb
    t0 = time.time()
    total_data_wrank_stats_tourney = merge_tourney(total_data_wrank_stats)
    print("[{0}s] 5) Merge with tourney database ".format(time.time() - t0))
    
    #### merge players info and replace missing values
    t0 = time.time()
    total_data_wrank_stats_tourney_players = merge_atp_players(total_data_wrank_stats_tourney)
    print("[{0}s] 6) Merge with players and fillin missing values ".format(time.time() - t0))
    print(pd.isnull(total_data_wrank_stats_tourney_players).sum())
    
    total_data_wrank_stats_tourney_players.rename(columns = {"surface_x": "surface", "tourney_name_x" : "tourney_name"}, inplace= True)
    
    return total_data_wrank_stats_tourney_players


def merge_atp_missing_stats(total_data, redo = False):
    
    data = total_data.copy()
    missing_match_stats= match_stats_main(data, redo = redo)
    
    for i in missing_match_stats["ATP_ID"].tolist():
        if pd.isnull(data.loc[data["ATP_ID"] == i, "w_ace"].values[0]):
            for col in ["w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",\
                        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced", "minutes"]:
                data.loc[data["ATP_ID"] == i, col] = missing_match_stats.loc[missing_match_stats["ATP_ID"] == i, col].values[0]
        else:
            data.loc[data["ATP_ID"] == i, "minutes"] = missing_match_stats.loc[missing_match_stats["ATP_ID"] == i, "minutes"].values[0]
   
    return data


def merge_tourney(data):
    
    tournament = pd.read_csv(os.environ["DATA_PATH"] + "/clean_datasets/tournament/tourney.csv", encoding = "latin1")
    tournament.loc[pd.isnull(tournament["masters"]), "masters"] = "classic"
    tournament.loc[pd.isnull(tournament["Currency"]), "Currency"] = "$"
    
    data_merge = pd.merge(data, tournament, on = "tourney_id", how = "left")
     
    data_merge = data_merge.drop(["tourney_name", "surface_y", "tourney_id_atp", "tourney_year"], axis=1)
    
    return data_merge
 
