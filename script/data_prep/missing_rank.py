# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:34:54 2018

@author: User
"""

import pandas as pd
import numpy as np

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
    

def fill_ranks_based_origin(total_data):
    
    data = total_data.copy()
    missing_data_rank = data.loc[(pd.isnull(data["winner_rank"]))|(pd.isnull(data["loser_rank"]))].copy()
    
    ### fillin missing ranks and points with closest previous rank and point
    missing_data_rank["id_rank_pts"]  = missing_data_rank[["Date", "winner_id", "loser_id", "winner_rank", "loser_rank"]].apply(lambda x : deduce_rank_from_past(x, data), axis=1)["loser_rank"]

    index_w = pd.isnull(data["winner_rank"])
    data.loc[index_w, "winner_rank"] = list(list(zip(*missing_data_rank.loc[pd.isnull(missing_data_rank["winner_rank"]), "id_rank_pts"]))[0])
    data.loc[index_w, "winner_rank_points"] = list(list(zip(*missing_data_rank.loc[pd.isnull(missing_data_rank["winner_rank"]), "id_rank_pts"]))[1])
    index_l = pd.isnull(data["loser_rank"])
    data.loc[index_l, "loser_rank"] = list(list(zip(*missing_data_rank.loc[pd.isnull(missing_data_rank["loser_rank"]), "id_rank_pts"]))[2])
    data.loc[index_l, "loser_rank_points"] = list(list(zip(*missing_data_rank.loc[pd.isnull(missing_data_rank["loser_rank"]), "id_rank_pts"]))[3])

    return data
