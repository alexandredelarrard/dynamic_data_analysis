# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:34:54 2018

@author: User
"""

import pandas as pd
import numpy as np
import glob
import os


def deduce_rank_from_past(x, data):

    id_missing_winner=  True if pd.isnull(x["winner_rank"]) else False
    id_missing_loser=  True if pd.isnull(x["loser_rank"]) else False
    
     ### the missing value comes from the winning player
    if id_missing_winner:
        sub_data = data.loc[((data["winner_id"] == x["winner_id"])&(~pd.isnull(data["winner_rank"]))) | ((data["loser_id"] == x["winner_id"]) &(~pd.isnull(data["loser_rank"])))].copy()
        sub_data["time_dist"] = abs((x["Date"] - sub_data["Date"]).dt.days)
        
        if len(sub_data) ==0: #### no previous or after value ---> this is a new person : take percentile 99 
             missed =[int(np.percentile(data.loc[~pd.isnull(data["loser_rank"]), "loser_rank"], q = 99)), 
                                        int(np.percentile(data.loc[~pd.isnull(data["loser_rank_points"]), "loser_rank_points"], q = 95))]
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
        
        if len(sub_data) ==0: #### no previous or after value ---> this is a new person : take percentile 99 
             missed +=[int(np.percentile(data.loc[~pd.isnull(data["loser_rank"]), "loser_rank"], q = 99)), 
                                          int(np.percentile(data.loc[~pd.isnull(data["loser_rank_points"]), "loser_rank_points"], q = 95))]
        else:
            elect = sub_data.loc[sub_data["time_dist"] == min(sub_data["time_dist"])].iloc[0]
            
            rank = elect["winner_rank"] if elect["winner_id"] == x["loser_id"] else elect["loser_rank"]
            pts  = elect["winner_rank_points"] if elect["winner_id"] == x["loser_id"] else elect["loser_rank_points"]
        
            missed +=[rank, pts]     
    else:
        missed += [np.nan, np.nan]
        
    return [missed]


def match_rank_missing(x, rk_data):
    """
    x : 0,1,2 = "Date","winner_name","loser_name"
    rk_data: 0,1,2,3 = name, rank, pts, date
    """
    data = rk_data[rk_data[:,3] <= x[0],:]
    
    sub_data_w = data[data[:,0] == x[1].lower().replace("-"," "),:]
    sub_data_l = data[data[:,0] == x[2].lower().replace("-"," "),:]
    
    if sub_data_w.shape[0] >0:
        rk_w, rk_pts_w = int(sub_data_w[-1][1].replace("T","")), sub_data_w[-1][2]
    else:
        rk_w, rk_pts_w = -1, -1
        
    if sub_data_l.shape[0] >0:
        rk_l, rk_pts_l = int(sub_data_l[-1][1].replace("T","")), sub_data_l[-1][2]
    else:
        rk_l, rk_pts_l = -1, -1
        
    return [(rk_w, rk_pts_w, rk_l, rk_pts_l)]


def deduce_rank_from_atp(total_data):
    data = total_data.copy()
    missing_data_rank = data.loc[(pd.isnull(data["winner_rank"]))|(pd.isnull(data["loser_rank"]))].copy()
    
    files_rank = glob.glob(os.environ["DATA_PATH"] + "/brute_info/atp_ranking/*.csv")
    files_df = pd.DataFrame(np.transpose([files_rank, [pd.to_datetime(os.path.splitext(os.path.basename(x))[0], format = "%Y-%m-%d") for x in files_rank]]), columns = ["file", "Date"])
    
    for i, f in enumerate(files_df["file"].tolist()):
        if i ==0:
            rk_data = pd.read_csv(f)
            rk_data["Date"] = files_df.iloc[i,1]
            rk_data = np.array(rk_data)
            
        else:
            new_data = pd.read_csv(f)
            new_data["Date"] = files_df.iloc[i,1]
            rk_data = np.concatenate((rk_data, np.array(new_data)), axis =0)
            
    rk_data = np.concatenate((rk_data[:,:3], rk_data[:,5:]), axis=1)
            
    rk_pts_missing = np.apply_along_axis(match_rank_missing, 1, np.array(missing_data_rank[["Date", "winner_name", "loser_name"]]), rk_data) 
    rk_pts_missing = rk_pts_missing.reshape(rk_pts_missing.shape[0], rk_pts_missing.shape[2])
    
    columns = ["winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points"]
    data.loc[(pd.isnull(data["winner_rank"]))|(pd.isnull(data["loser_rank"])), columns] = rk_pts_missing
    
    return data 


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
