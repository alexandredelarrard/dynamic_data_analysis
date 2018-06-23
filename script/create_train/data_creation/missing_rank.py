# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:34:54 2018

@author: User
"""

import pandas as pd
import numpy as np

from create_train.utils.utils_data_prep import import_rank_data

def match_rank_missing(x, rk_data):
    """
    x : 0,1,2 = "Date","winner_name","loser_name"
    rk_data: 0,1,2,3 = name, rank, pts, date
    """
    data = rk_data[rk_data[:,3] <= x[0],:]
    
    sub_data_w = data[data[:,0] == x[1].lower().replace("-"," "),:]
    sub_data_l = data[data[:,0] == x[2].lower().replace("-"," "),:]
    
    if sub_data_w.shape[0] >0:
        rk_w, rk_pts_w = sub_data_w[-1][1], sub_data_w[-1][2]
    else:
        rk_w, rk_pts_w = -1, -1
        
    if sub_data_l.shape[0] >0:
        rk_l, rk_pts_l = sub_data_l[-1][1], sub_data_l[-1][2]
    else:
        rk_l, rk_pts_l = -1, -1
        
    return [(rk_w, rk_pts_w, rk_l, rk_pts_l)]


def deduce_rank_from_atp(total_data):
    data = total_data.copy()
    missing_data_rank = data.loc[(pd.isnull(data["winner_rank"]))|(pd.isnull(data["loser_rank"]))].copy()
    
    rk_data = import_rank_data(data)
    rk_data = np.array(rk_data[["Player_name", "player_rank", "player_points", "Date"]])
            
    rk_pts_missing = np.apply_along_axis(match_rank_missing, 1, np.array(missing_data_rank[["Date", "winner_name", "loser_name"]]), rk_data) 
    rk_pts_missing = rk_pts_missing.reshape(rk_pts_missing.shape[0], rk_pts_missing.shape[2])
    
    columns = ["winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points"]
    rk_pts_missing = pd.DataFrame(rk_pts_missing, columns = columns)
    rk_pts_missing.index = missing_data_rank.index
    data.loc[(pd.isnull(data["winner_rank"]))|(pd.isnull(data["loser_rank"])), columns] = rk_pts_missing
    
    # =============================================================================
    #     #### handle missing values as no rank so 1800 (wors as being new)
    # =============================================================================
    data.loc[data["winner_rank"]<0, ["winner_rank", "winner_rank_points"]] = 1800,0
    data.loc[data["loser_rank"]<0, ["loser_rank", "loser_rank_points"]] = 1800,0
    
    # =============================================================================
    #     #### before 1996-08-12 all points are null ----> fill in with average pts same rank 
    # =============================================================================
#    data["weeks"] = data["Date"].dt.week
#    data_futur = data.loc[(data["Date"]>="1996-08-12")&(data["Date"]<= "2003-12-31")].copy()
#    data_futur = pd.concat([pd.DataFrame(np.array(data_futur[["winner_rank", "winner_rank_points", "weeks"]])), 
#                            pd.DataFrame(np.array(data_futur[["loser_rank", "loser_rank_points", "weeks"]]))], axis=0)
#    data_futur.columns = ["rank","rank_points","week"]
#    aggregation = data_futur[["rank","rank_points","week"]].groupby(["rank","week"]).mean().reset_index()
#    
#    for player in ["winner", "loser"]:
#        fillin_rank_pts_w = data.loc[(data["%s_rank_points"%player] == 0)&(data["Date"]<"1996-08-12"), ["%s_rank"%player, "weeks"]]
#        fillin_rank_pts_w = pd.merge(fillin_rank_pts_w, aggregation, left_on = ["weeks", "%s_rank"%player], right_on = ["week", "rank"], how="left")
#        data.loc[(data["%s_rank_points"%player] == 0)&(data["Date"]<"1996-08-12"), "%s_rank_points"%player] = fillin_rank_pts_w["rank_points"]
#        
#    print(pd.isnull(data).sum())
    return data 
