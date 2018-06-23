# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 10:29:47 2018

@author: User
"""

import pandas as pd
import time 
import numpy as np
import os
from datetime import datetime

def walkover_retired(x, data):
    """
    columns : ["ref_days", "winner_id", "loser_id", "status"]
    """

    data_sub = data[np.where((data[:,0] < x[0]))]
    
    winner_data = data_sub[np.where(data_sub[:,2] == x[2])]
    loser_data = data_sub[np.where(data_sub[:,2] == x[1])]
    
    winner_data_inj = winner_data[np.where(np.isin(winner_data[:,3], ["Retired", "Walkover", "Def"]))]
    loser_data_inj = winner_data[np.where(np.isin(winner_data[:,3], ["Retired", "Walkover", "Def"]))]
    
    nbr_injuries_w = winner_data_inj.shape[0]
    nbr_injuries_l = loser_data_inj.shape[0]
    
    if nbr_injuries_w> 0:
        time_inj = winner_data_inj[:,0].max()
        delta_match_last_injury_w = winner_data[np.where(winner_data[:,0]> time_inj)].shape[0]
        
        if delta_match_last_injury_w == 0: #### first match since injury
            last_injury_last_w = x[0] - time_inj
        else:
            last_injury_last_w = winner_data[np.where(winner_data[:,0]> time_inj)][0,0] - time_inj
            
    else:
        last_injury_last_w = 0
        delta_match_last_injury_w = -1
        
    if nbr_injuries_l> 0:
        time_inj = loser_data_inj[:,0].max()
        delta_match_last_injury_l = loser_data[np.where(loser_data[:,0]> time_inj)].shape[0]
        
        if delta_match_last_injury_l == 0:
            last_injury_last_l = x[0] - time_inj
        else:
            last_injury_last_l = loser_data[np.where(loser_data[:,0]> time_inj)][0,0] - time_inj
          
    else:
        last_injury_last_l = 0
        delta_match_last_injury_l = -1
    
    return (nbr_injuries_w, nbr_injuries_l, last_injury_last_w, last_injury_last_l, delta_match_last_injury_w, delta_match_last_injury_l)


def return_after_injury(new_data, data):
    
    t0 = time.time()
    new_data["ref_days"]= (new_data["Date"]- pd.to_datetime("01/01/1901")).dt.days
    data["ref_days"]= (data["Date"]- pd.to_datetime("01/01/1901")).dt.days
    
    injury = np.apply_along_axis(walkover_retired, 1, np.array(new_data[["ref_days", "winner_id", "loser_id"]]), np.array(data[["ref_days", "winner_id", "loser_id", "status"]]))
    injury = pd.DataFrame(injury, columns = ["nbr_injuries_w", "nbr_injuries_l", "last_injury_last_w", "last_injury_last_l", "delta_match_last_injury_w", "delta_match_last_injury_l"])    
    new_data = pd.concat([new_data, injury], axis = 1)
    del new_data["ref_days"]
    del data["ref_days"]
    print(" --- calculate return after walkover or retired : {0} ".format(time.time() - t0))
    
    ### suppress not completed match and save injuries to wound folder
    if new_data.shape[0] == data.shape[0]:
        path = os.environ["DATA_PATH"] + "/clean_datasets/wounds/history/injury_history.csv"
    else:
        path = os.environ["DATA_PATH"] + "/clean_datasets/wounds/extracted/injury_history_{0}.csv".format(datetime.now().strftime("%Y-%m-%d"))
    
    new_data.loc[new_data["status"].isin(["Retired", "Walkover", "Def"])][["Date", "tourney_id", "winner_id", "loser_id", "status"]].to_csv(path, index= False)
    new_data = new_data.loc[~new_data["status"].isin(["Retired", "Walkover", "Def"])]
    
    return new_data



def extract_rank_and_match(x, rk_data):
    """
    match rank with player name loser and winner based on closest date into the past
    """
    
    dates = pd.to_datetime(rk_data.sort_values("Date")["Date"].unique())
    date = dates[dates <= x["Date"]][-1]
    rank_sub_df = rk_data.loc[rk_data["Date"] == date]
    
    try:
        winner = rank_sub_df.loc[rank_sub_df["Player_name"] == x["winner_name"]][["player_rank", "player_points"]].values[0]
    except Exception:
        print(x)
        print(rank_sub_df.loc[rank_sub_df["Player_name"] == x["loser_name"]][["player_rank", "player_points"]])
        winner = [1800, 0]    
        
    try:
        loser =  rank_sub_df.loc[rank_sub_df["Player_name"] == x["loser_name"]][["player_rank", "player_points"]].values[0]
    except Exception:
        print(x)
        print(rank_sub_df.loc[rank_sub_df["Player_name"] == x["loser_name"]][["player_rank", "player_points"]])
        loser = [1800, 0]
        
    return [(winner[0],winner[1],loser[0],loser[1])]