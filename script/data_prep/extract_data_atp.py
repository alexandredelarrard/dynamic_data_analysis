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
    
    data.loc[data["winner_name"] == "joshua goodall", "winner_name"] = "josh goodall"
    data.loc[data["loser_name"] == "joshua goodall", "loser_name"] = "josh goodall"
    data = data.reset_index(drop=True)
    print("[{0}s] 1) Import ATP dataset ".format(time.time() - t0))
    
    total_data = fill_in_missing_values(data, redo)
            
    return total_data


def fill_in_missing_values(total_data, redo):
    
    #### suppress davis cup and JO
    not_davis_index = total_data["tourney_name"].apply(lambda x : "Davis Cup" not in x and "Olympic" not in x)
    total_data = total_data.loc[not_davis_index]
    
    #### fill in missing scores
    total_data.loc[(total_data["tourney_id"] == "2007-533")&(pd.isnull(total_data["score"])), "score"] = "6-1 4-6 7-5"
    total_data.loc[(total_data["tourney_id"] == "1997-319")&(pd.isnull(total_data["score"])), "score"] = "6-4 6-4 6-4"
    
    #### fill in missing ranks and points
#    t0 = time.time()
#    total_data_wrank = fill_ranks_based_origin(total_data)
#    total_data_wrank = total_data_wrank.drop(["winner_seed", "winner_entry", "loser_seed", "loser_entry"],axis=1)
#    print("[{0}s] 2) fill missing rank based on closest info ".format(time.time() - t0))
    
    #### add match stats on service missing
    t0 = time.time()
    total_data_wrank_stats = merge_atp_missing_stats(total_data, redo)
    print("[{0}s] 3) fill missing stats based on atp crawling matching ".format(time.time() - t0))
    
    #### merge with tourney ddb
    t0 = time.time()
    total_data_wrank_stats_tourney = merge_tourney(total_data_wrank_stats)
    print("[{0}s] 4) Merge with tourney database ".format(time.time() - t0))
    
    #### merge players info and replace missing values
    t0 = time.time()
    total_data_wrank_stats_tourney_players = merge_atp_players(total_data_wrank_stats_tourney)
    print("[{0}s] 5) Merge with players and fillin missing values ".format(time.time() - t0))
    print(pd.isnull(total_data_wrank_stats_tourney_players).sum())
    
    return total_data_wrank_stats_tourney_players


def fill_ranks_based_origin(total_data):
    
    data = total_data.copy()
    missing_data_rank = data.loc[(pd.isnull(data["winner_rank"]))|(pd.isnull(data["loser_rank"]))].copy()
    
    ### fillin missing ranks and points with closest previous rank and point
    missing_data_rank["id_rank_pts"]  = missing_data_rank[["Date", "winner_id", "loser_id", "winner_rank", "loser_rank"]].apply(lambda x : deduce_rank_from_past(x, total_data), axis=1)["loser_rank"]

    index_w = pd.isnull(data["winner_rank"])
    data.loc[index_w, "winner_rank"] = list(list(zip(*missing_data_rank.loc[pd.isnull(missing_data_rank["winner_rank"]), "id_rank_pts"]))[0])
    data.loc[index_w, "winner_rank_points"] = list(list(zip(*missing_data_rank.loc[pd.isnull(missing_data_rank["winner_rank"]), "id_rank_pts"]))[1])
    index_l = pd.isnull(data["loser_rank"])
    data.loc[index_l, "loser_rank"] = list(list(zip(*missing_data_rank.loc[pd.isnull(missing_data_rank["loser_rank"]), "id_rank_pts"]))[2])
    data.loc[index_l, "loser_rank_points"] = list(list(zip(*missing_data_rank.loc[pd.isnull(missing_data_rank["loser_rank"]), "id_rank_pts"]))[3])

    return data


def deduce_rank_from_past(x, data):

    id_missing_winner=  True if pd.isnull(x["winner_rank"]) else False
    id_missing_loser=  True if pd.isnull(x["loser_rank"]) else False
    
     ### the missing value comes from the winning player
    if id_missing_winner:
        sub_data = data.loc[((data["winner_id"] == x["winner_id"])&(~pd.isnull(data["winner_rank"]))) | ((data["loser_id"] == x["winner_id"]) &(~pd.isnull(data["loser_rank"])))].copy()
        sub_data["time_dist"] = abs((x["Date"] - sub_data["Date"]).dt.days)
        
        if len(sub_data) ==0:
             missed =[ int(data["winner_rank"].mean()), int(data["winner_rank_points"].mean())]
        else:
            elect = sub_data.loc[sub_data["time_dist"] == min(sub_data["time_dist"])].iloc[0]
            
            rank = elect["winner_rank"] if elect["winner_id"] == x["winner_id"] else elect["loser_rank"]
            pts  = elect["winner_rank_points"] if elect["winner_id"] == x["winner_id"] else elect["loser_rank_points"]
        
            missed =[rank, pts]
    else:
        missed = [np.nan, np.nan]
    
    ### the missing value comes from the losing player
    if id_missing_loser:
        sub_data = data.loc[((data["winner_id"] == x["loser_id"])&(~pd.isnull(data["winner_rank"]))) | ((data["loser_id"] == x["loser_id"]) &(~pd.isnull(data["loser_rank"])))].copy()
        sub_data["time_dist"] = abs((x["Date"] - sub_data["Date"]).dt.days)
        
        if len(sub_data) ==0:
             missed +=[ int(data["loser_rank"].mean()), int(data["loser_rank_points"].mean())]
        else:
            elect = sub_data.loc[sub_data["time_dist"] == min(sub_data["time_dist"])].iloc[0]
            
            rank = elect["winner_rank"] if elect["winner_id"] == x["loser_id"] else elect["loser_rank"]
            pts  = elect["winner_rank_points"] if elect["winner_id"] == x["loser_id"] else elect["loser_rank_points"]
        
            missed +=[rank, pts]     
    else:
        missed += [np.nan, np.nan]
        
    return [missed]
    

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
 
