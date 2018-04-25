# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 13:38:19 2018

@author: JARD
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from multiprocessing import Pool
from functools import partial
from datetime import  timedelta

from utils.weight_past_matches import calculate_corr_surface, calculate_corr_time, calculate_corr_opponents

def parallelize_dataframe(df, function, dictionnary, njobs):
    df_split = np.array_split(df, njobs)
    pool = Pool(njobs)
    func = partial(function, dictionnary)
    df2 = pd.concat(pool.map(func, df_split))
    
    pool.close()
    pool.join()
    
    return df2

def common_opponents(x, data):
    
    oppo_win  = data.loc[data["winner_id"] == x["winner_id"], "loser_id"].tolist() +  data.loc[data["loser_id"] == x["winner_id"], "winner_id"].tolist()
    oppo_lose = data.loc[data["loser_id"] == x["loser_id"], "winner_id"].tolist() +  data.loc[data["winner_id"] == x["loser_id"], "loser_id"].tolist() 
    
    interset_ids = list(set.intersection(set(oppo_win), set(oppo_lose)))
    players = [x["winner_id"], x["loser_id"]]
    
    sub_data = data.loc[((data["winner_id"].isin(players)) & (data["loser_id"].isin(interset_ids + players)))|
                        ((data["loser_id"].isin(players)) & (data["winner_id"].isin(interset_ids + players)))]
    
    return sub_data


def add_weight(x, sub_data, corr_surface, corr_time):
    
    def diff_month(d1, d2):
        return (d1.year - d2.dt.year) * 12 + d1.month - d2.dt.month
    
    sub_data["weight"] = list(diff_month(x["Date"], sub_data["Date"]))
    sub_data["weight"] = sub_data["surface"].map(corr_surface[x["surface"]])*sub_data["weight"].map(corr_time)
    
    return sub_data


def get_stats(x, sub_data):
    
    winner_w_data = sub_data.loc[sub_data["winner_id"] == x["winner_id"]]
    winner_l_data = sub_data.loc[sub_data["loser_id"] == x["winner_id"]]
    loser_w_data = sub_data.loc[sub_data["winner_id"] == x["loser_id"]]
    loser_l_data = sub_data.loc[sub_data["loser_id"] == x["loser_id"]]
    
    weight_winner = (winner_w_data["weight"].sum() + winner_l_data["weight"]).sum()
    weight_loser = (loser_w_data["weight"].sum() + loser_l_data["weight"]).sum()
    
    count = (sub_data.shape[0], ### confidence on stat
             
             (winner_w_data["w_ace"]*winner_w_data["weight"]  + winner_l_data["l_ace"]*winner_l_data["weight"]).sum()/weight_winner -\
             (loser_w_data["w_ace"]*loser_w_data["weight"]  + loser_l_data["l_ace"]*loser_l_data["weight"]).sum()/weight_loser, #### difference aces
             
             
             )
    
    return count
    

def weighted_statistics(x, liste_dataframe):
    
    data = liste_dataframe[0]
    corr_surface = liste_dataframe[1]
    corr_time = liste_dataframe[2]
    
    sub_data = common_opponents(x, data.loc[data["Date"]< x["Date"]])
    sub_data = add_weight(x, sub_data, corr_surface, corr_time)
    
    stats    = get_stats(x, sub_data)
    
    return stats
    

def global_stats(data):
    
    data["diff_age"] = ((data["Date"] - data["DOB_w"]).dt.days - (data["Date"] - data["DOB_l"]).dt.days)/365
    data["diff_ht"] = data["winner_ht"] - data["loser_ht"]
    data["diff_weight"] = data["Weight_w"] - data["Weight_l"]
    data["diff_year_turned_pro"] = data['Turned pro_w'] - data['Turned pro_l']
    
    data["diff_elo"] = data['elo1'] - data['elo2']
    data["diff_rank"] = data['winner_rank'] - data['loser_rank']
    data["diff_rk_pts"] = data['winner_rank_points'] - data['loser_rank_points']
    data["diff_hand"] = data['winner_hand'] - data['loser_hand']
    
    return data


def create_statistics(data):
    
    data = global_stats(data)
    
    data1 = data[["Date", "winner_id", "loser_id", "surface", "tourney_id"]].copy()
    data1["target"] = 1
    
    data2 = data1.copy()
    data2 = data2.rename(columns = {"winner_id" : "loser_id", "loser_id" : "winner_id"})
    data2["target"] = 0
    data2 = data2[["winner_id", "loser_id", "Date", "target", "surface", "tourney_id"]]
    
    tot = pd.concat([data1, data2], axis= 0)
    tot["Date"] = pd.to_datetime(tot["Date"], format = "%Y-%m-%d")
    
    correlation_surface   = calculate_corr_surface(tot, start_year=1990, end_year=2017)
    correlation_time      = calculate_corr_time(tot, start_year=1990, end_year=2017)
#    correlation_opponents = calculate_corr_opponents(tot)
    
    col_for_stats = ['Date', 'winner_id', 'loser_id', 'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
                     'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced','w_1st_srv_ret_won',
                     'w_2nd_srv_ret_won', 'w_bp_converted', 'w_total_srv_won', 'w_total_ret_won', 'l_1st_srv_ret_won', 'l_2nd_srv_ret_won', 'l_bp_converted',
                     'l_total_srv_won', 'l_total_ret_won', 'w_tie-breaks_won', 'l_tie-breaks_won', 'Nbr_tie-breaks']
    
    counts = data1.apply(lambda x : weighted_statistics(x, [data[col_for_stats], correlation_surface, correlation_time]), axis= 1)
    
    return tot


if __name__ == "__mane__": 
    
    data = pd.read_csv(r"C:\Users\User\Documents\tennis\data\clean_datasets\historical\matches_elo_variables_V1.csv")
    data["Date"] = pd.to_datetime(data["Date"], format = "%Y-%m-%d")
    data["DOB_w"] = pd.to_datetime(data["DOB_w"], format = "%Y-%m-%d")
    data["DOB_l"] = pd.to_datetime(data["DOB_l"], format = "%Y-%m-%d")
    tot = create_statistics(data)
    