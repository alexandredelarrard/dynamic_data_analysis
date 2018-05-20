# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 14:00:49 2018

@author: User
"""

import pandas as pd
import glob
import time 
import numpy as np

from create_data.utils.build_match_statistics_database import match_stats_main
from create_data.data_creation.extract_players import  merge_atp_players
from create_data.data_creation.missing_rank  import fill_ranks_based_origin
from create_data.data_creation.missing_stats  import fillin_missing_stats
from create_data.data_creation.merge_tourney  import merge_tourney


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
    
    ### identify a game as retired:  Andreas Seppi  Mario Ancic 2007-02-12    Marseille
    data.loc[(data["Date"] == "2007-02-12")&(data["winner_name"] == "Andreas Seppi")&(data["loser_name"] == "Mario Ancic"), "score"] = "RET"
    
    #### suppress davis cup and JO
    sp = data.shape[0] 
    not_davis_index = data["tourney_name"].apply(lambda x : "Davis Cup" not in x and "Olympic" not in x)
    data = data.loc[not_davis_index]
    print(" --- Suppress davis cup and JO : {0} ".format(sp - data.shape[0]))
    
    #### suppress challenger matches
    sp = data.shape[0] 
    data = data.loc[data["tourney_level"] != "C"]
    print(" --- Suppress challenger matches : {0} ".format(sp - data.shape[0]))
    
    ### create status of match
    data["status"] = "Completed"
    index = data["score"].apply(lambda x : "RET" in str(x))
    data.loc[index, "status"] = "Retired"
    index = data["score"].apply(lambda x : "DEF" in str(x) or "Jun" in str(x))
    data.loc[index, "status"] = "Def"
    index = data["score"].apply(lambda x : "W/O" in str(x) or "W/O " in str(x) or " W/O" in str(x))
    data.loc[index, "status"] = "Walkover"
    
    #### create variable nbr days since retired / walkover
    
    t0 = time.time()
    data["ref_days"]= (data["Date"]- pd.to_datetime("01/01/1901")).dt.days
    data["diff_days_since_stop"] = np.apply_along_axis(walkover_retired, 1, np.array(data[["ref_days", "winner_id", "loser_id"]]), np.array(data[["ref_days", "winner_id", "loser_id", "status"]]))
    del data["ref_days"]
    print(" --- calculate return after walkover or retired : {0} ".format(time.time() - t0))

    ### suppress not completed match
    data = data.loc[~data["status"].isin(["Retired", "Walkover", "Def"])]
    
    #### fill in missing scores
    data.loc[(data["tourney_id"] == "2007-533")&(pd.isnull(data["score"])), "score"] = "6-1 4-6 7-5"
    data.loc[(data["tourney_id"] == "1997-319")&(pd.isnull(data["score"])), "score"] = "6-4 6-4 6-4"
    
    data.loc[data["winner_name"] == "joshua goodall", "winner_name"] = "josh goodall"
    data.loc[data["loser_name"] == "joshua goodall", "loser_name"] = "josh goodall"
    data = data.reset_index(drop=True)
    print("\n [{0}s] 1) Import ATP dataset ".format(time.time() - t0))

    total_data = fill_in_missing_values(data, redo)
            
    return total_data


def fill_in_missing_values(total_data, redo):
    
    mvs = pd.isnull(total_data).sum()
    #### fill in missing ranks and points
    t0 = time.time()
    total_data_wrank = fill_ranks_based_origin(total_data)
    total_data_wrank = total_data_wrank.drop(["winner_seed", "winner_entry", "loser_seed", "loser_entry"],axis=1)
    print("[{0}s] 2) fill missing rank based on closest info ({1}/{2})".format(time.time() - t0, mvs["loser_rank"] + mvs["winner_rank"], total_data.shape[0]))
    
    #### add match stats missing
    t0 = time.time()
    total_data_wrank_stats = merge_atp_missing_stats(total_data_wrank, redo)
    print("[{0}s] 3) fill missing stats based on atp crawling matching  ({1}/{2})".format(time.time() - t0, mvs["w_ace"], total_data.shape[0]))
    
    #### fill in irreductible missing values based on history
    mvs = pd.isnull(total_data_wrank_stats).sum()
    t0 = time.time()
    total_data_wrank_stats = fillin_missing_stats(total_data_wrank_stats)
    print("[{0}s] 4) fill missing stats based on previous matches ({1}/{2})".format(time.time() - t0, mvs["w_ace"], total_data.shape[0]))
    
    #### merge with tourney ddb
    t0 = time.time()
    total_data_wrank_stats_tourney = merge_tourney(total_data_wrank_stats)
    print("[{0}s] 5) Merge with tourney database ".format(time.time() - t0))
    
    #### merge players info and replace missing values
    t0 = time.time()
    total_data_wrank_stats_tourney_players = merge_atp_players(total_data_wrank_stats_tourney)
    print("[{0}s] 6) Merge with players and fillin missing values (ht : {1}/{2}; age: {3}/{2})".format(time.time() - t0, mvs["winner_ht"] + mvs["loser_ht"], total_data.shape[0], mvs["winner_age"] + mvs["loser_age"]))
    print(pd.isnull(total_data_wrank_stats_tourney_players).sum())
    
    total_data_wrank_stats_tourney_players = total_data_wrank_stats_tourney_players.rename(columns = {"surface_x": "surface", "tourney_name_x" : "tourney_name"})
    
    #### remaining mvs for time of match
    total_data_wrank_stats_tourney_players.loc[pd.isnull(total_data_wrank_stats_tourney_players["minutes"]), "minutes"] = 90
    
    return total_data_wrank_stats_tourney_players


def merge_atp_missing_stats(total_data, redo = False):
    
    data = total_data.copy()
    missing_match_stats= match_stats_main(data, redo = redo)
    
    for i in missing_match_stats["ATP_ID"].tolist():
        try:
            if pd.isnull(data.loc[data["ATP_ID"] == i, "w_ace"].values[0]):
                for col in ["w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",\
                            "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced", "minutes"]:
                    data.loc[data["ATP_ID"] == i, col] = missing_match_stats.loc[missing_match_stats["ATP_ID"] == i, col].values[0]
            else:
                data.loc[data["ATP_ID"] == i, "minutes"] = missing_match_stats.loc[missing_match_stats["ATP_ID"] == i, "minutes"].values[0]
        except Exception:
           data.loc[data["ATP_ID"] == i, "w_ace"]
    return data


def walkover_retired(x, data):
    """
    columns : ["ref_days", "winner_id", "loser_id", "status"]
    """

    index = np.where((data[:,0] < x[0])&(np.isin(data[:,3], ["Retired", "Walkover", "Def"])))
    data_sub = data[index]
    
    index_l = np.where(data_sub[:,2] == x[2])
    index_w = np.where(data_sub[:,2] == x[1])
    
    if len(index_l[0])>0:
        ref_day = np.max(data_sub[index_l, 0])
        l = 1 if (x[0] - ref_day) <= 10 else 0
    else:
        l=0
          
    if len(index_w[0])>0:   
        ref_day = np.max(data_sub[index_w, 0])
        w = 1 if (x[0] - ref_day)  <= 10 else 0
    else:
        w=0
        
    return w - l

      
