# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 13:38:19 2018

@author: JARD
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial
import sys
import time
from tqdm import tqdm

tqdm.pandas(tqdm())

sys.path.append(r"C:\Users\User\Documents\tennis\dynamic_data_analysis\script")
from utils.weight_past_matches import calculate_corr_surface, calculate_corr_time


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
    
    try:
        sub_data["weight"] = sub_data["surface"].map(corr_surface[x["surface"]])*sub_data["weight"].map(corr_time)
        
    except Exception:
        print(x)
        print(corr_surface[x["surface"]])
        print(sub_data["surface"])
        
    return sub_data


def weighted_statistics(x, liste_dataframe):
    
    data = liste_dataframe[0].copy()
    corr_surface = liste_dataframe[1]
    corr_time = liste_dataframe[2]
    
    sub_data = common_opponents(x, data.loc[data["Date"] < x["Date"]])
    sub_data = add_weight(x, sub_data, corr_surface, corr_time)
    
    stats    = get_stats(x, sub_data)
    return stats


def get_stats(x, sub_data):
    
    winner_w_data = sub_data.loc[sub_data["winner_id"] == x["winner_id"]]
    winner_l_data = sub_data.loc[sub_data["loser_id"] == x["winner_id"]]
    loser_w_data = sub_data.loc[sub_data["winner_id"] == x["loser_id"]]
    loser_l_data = sub_data.loc[sub_data["loser_id"] == x["loser_id"]]
    
    weight_winner = winner_w_data["weight"].sum() + winner_l_data["weight"].sum()
    weight_loser = loser_w_data["weight"].sum() + loser_l_data["weight"].sum()
    
    ws1 = (((winner_w_data["w_1stIn"]*winner_w_data["w_1stWon"] + (1-winner_w_data["w_1stIn"])*winner_w_data["w_2ndWon"])*winner_w_data["weight"]).sum() 
             + ((winner_l_data["l_1stIn"]*winner_l_data["l_1stWon"] + (1-winner_l_data["l_1stIn"])*winner_l_data["l_2ndWon"])*winner_l_data["weight"]).sum())/weight_winner 
    
    ws2 = (((loser_w_data["w_1stIn"]*loser_w_data["w_1stWon"] + (1-loser_w_data["w_1stIn"])*loser_w_data["w_2ndWon"])*loser_w_data["weight"]).sum()
             + ((loser_l_data["l_1stIn"]*loser_l_data["l_1stWon"] + (1-loser_l_data["l_1stIn"])*loser_l_data["l_2ndWon"])*loser_l_data["weight"]).sum())/weight_loser 
    
    wr1 = (((winner_w_data["w_1st_srv_ret_won"]*winner_w_data["l_1stIn"] + (1-winner_w_data["l_1stIn"])*winner_w_data["w_2nd_srv_ret_won"])*winner_w_data["weight"]).sum() 
             + ((winner_l_data["l_1st_srv_ret_won"]*winner_l_data["w_1stIn"] + (1-winner_l_data["w_1stIn"])*winner_l_data["l_2nd_srv_ret_won"])*winner_l_data["weight"]).sum())/weight_winner 
    
    wr2 = (((loser_w_data["w_1st_srv_ret_won"]*loser_w_data["l_1stIn"] + (1-loser_w_data["l_1stIn"])*loser_w_data["w_2nd_srv_ret_won"])*loser_w_data["weight"]).sum() 
             + ((loser_l_data["l_1st_srv_ret_won"]*loser_l_data["w_1stIn"] + (1-loser_l_data["w_1stIn"])*loser_l_data["l_2nd_srv_ret_won"])*loser_l_data["weight"]).sum())/weight_loser
    
    if sub_data.shape[0]> 0:
        count = (sub_data.shape[0], ### confidence on stat
                 
                 ((winner_w_data["w_ace"]*winner_w_data["weight"]).sum()  + (winner_l_data["l_ace"]*winner_l_data["weight"]).sum())/weight_winner -\
                 ((loser_w_data["w_ace"]*loser_w_data["weight"]).sum()  + (loser_l_data["l_ace"]*loser_l_data["weight"]).sum())/weight_loser, #### difference proportion aces
                 
                 ((winner_w_data["w_df"]*winner_w_data["weight"]).sum()  + (winner_l_data["l_df"]*winner_l_data["weight"]).sum())/weight_winner -\
                 ((loser_w_data["w_df"]*loser_w_data["weight"]).sum()  + (loser_l_data["l_df"]*loser_l_data["weight"]).sum())/weight_loser, #### difference proportion df
                 
                 ((winner_w_data["w_1stIn"]*winner_w_data["weight"]).sum()  + (winner_l_data["l_1stIn"]*winner_l_data["weight"]).sum())/weight_winner -\
                 ((loser_w_data["w_1stIn"]*loser_w_data["weight"]).sum()  + (loser_l_data["l_1stIn"]*loser_l_data["weight"]).sum())/weight_loser, #### difference proportion first serv
                 
                 ((winner_w_data["w_1stWon"]*winner_w_data["weight"]).sum()  + (winner_l_data["l_1stWon"]*winner_l_data["weight"]).sum())/weight_winner -\
                 ((loser_w_data["w_1stWon"]*loser_w_data["weight"]).sum()  + (loser_l_data["l_1stWon"]*loser_l_data["weight"]).sum())/weight_loser, #### difference proportion first won
                 
                 ((winner_w_data["w_2ndWon"]*winner_w_data["weight"]).sum()  + (winner_l_data["l_2ndWon"]*winner_l_data["weight"]).sum())/weight_winner -\
                 ((loser_w_data["w_2ndWon"]*loser_w_data["weight"]).sum()  + (loser_l_data["l_2ndWon"]*loser_l_data["weight"]).sum())/weight_loser, #### difference proportion second won
                 
                 #### overall skill on serv =  w1sp*fs + (1-fs)*w2sp
                 ws1 - ws2, 
                
                 ### overall skill on return = w1_ret* (l_fs) + (1- l_fs)*w2_ret
                 wr1 - wr2, 
                 
                 ### overall skill on both
                 ws1*ws1 - ws2*wr2,
                 
                 ### difference serv return 
                 ws1 - wr2,
                 
                 ### difference serv return 2
                 ws2- wr1,
                 
                 ### break point competencies  = bp_saved * bp_converted
                 ((winner_w_data["w_bpSaved"]*winner_w_data["w_bp_converted"]*winner_w_data["weight"]).sum() + (winner_l_data["l_bpSaved"]*winner_l_data["l_bp_converted"]*winner_l_data["weight"]).sum())/weight_winner -\
                 ((loser_w_data["w_bpSaved"]*loser_w_data["w_bp_converted"]*loser_w_data["weight"]).sum()  + (loser_l_data["l_bpSaved"]*loser_l_data["l_bp_converted"]*loser_l_data["weight"]).sum())/weight_loser, 
                 
                 ### tie break competencies 
                 ((winner_w_data["w_tie-breaks_won"]*winner_w_data["weight"]/winner_w_data["N_set"]).sum() + (winner_l_data["l_tie-breaks_won"]*winner_l_data["weight"]/winner_l_data["N_set"]).sum())/weight_winner -  
                 ((loser_w_data["w_tie-breaks_won"]*loser_w_data["weight"]/loser_w_data["N_set"]).sum() + (loser_l_data["l_tie-breaks_won"]*loser_l_data["weight"]/loser_l_data["N_set"]).sum())/weight_loser,
                 
                 ### proportion victory 1 vs 2 
                 (sub_data.loc[(sub_data["winner_id"] == x["winner_id"]) & (sub_data["loser_id"] == x["loser_id"])].shape[0] - 
                 sub_data.loc[(sub_data["winner_id"] == x["loser_id"]) & (sub_data["loser_id"] == x["winner_id"])].shape[0])/ sub_data.shape[0], 
                 
                  ### proportion victory common adversories
                 (sub_data.loc[(sub_data["winner_id"] == x["winner_id"])].shape[0] - 
                  sub_data.loc[(sub_data["winner_id"] == x["loser_id"])].shape[0])/ sub_data.shape[0],
                 
                 ### proportion points won 1 vs 2 
#                 sub_data.loc[(sub_data["winner_id"] == x["winner_id"]) & (sub_data["loser_id"] == x["loser_id"])].shape[0]/ sub_data.shape[0] 
                 )
    else:
          count = (0, )   + (np.nan,)*14
    
    return [count]
    


def execute_stats(wrong_word_dict, data):
    count = data.apply(lambda x: weighted_statistics(x, wrong_word_dict))
    return count

def fatigue_games(x , data):
    
    days= 4

    sub_data = data.loc[(x["ref_days"] - data["ref_days"]).between(1,days)]
    
    ### number of games played during last days days
    sub_data1 = sub_data.loc[((sub_data["winner_id"] == x["winner_id"]) | (sub_data["loser_id"] == x["winner_id"]))]
    fatigue_w = sub_data1["total_games"].sum()
    
    sub_data2 = sub_data.loc[((sub_data["winner_id"] == x["loser_id"]) | (sub_data["loser_id"] == x["loser_id"]))]
    fatigue_l = sub_data2["total_games"].sum()
    
    return fatigue_w - fatigue_l


def total_score(x):
    try:
        return sum([int(i.replace("[","").replace("]","").replace("'","")) for i in x if i !=""])
    except Exception:
        print(x)
        return 0

def global_stats(data):
    
    data = data.copy()
    data["diff_age"] = ((data["Date"] - data["DOB_w"]).dt.days - (data["Date"] - data["DOB_l"]).dt.days)/365
    data["diff_ht"] = data["winner_ht"] - data["loser_ht"]
    data["diff_weight"] = data["Weight_w"] - data["Weight_l"]
    data["diff_year_turned_pro"] = data['Turned pro_w'] - data['Turned pro_l']
    
    data["diff_elo"] = data['elo1'] - data['elo2']
    data["diff_rank"] = data['winner_rank'] - data['loser_rank']
    data["diff_rk_pts"] = data['winner_rank_points'] - data['loser_rank_points']
    data["diff_hand"] = data['winner_hand'] - data['loser_hand']
    data["diff_is_birthday"] = data['w_birthday'] - data['l_birthday']
    data["diff_home"] = data['w_home'] - data['l_home']
    
    return data


def create_statistics(data, redo = False):
    
    if redo :
        #### calculate correlations
        data1 = data[["Date", "winner_id", "loser_id", "surface", "tourney_id"]].copy()
        data1["target"] = 1
        
        data2 = data1.copy()
        data2 = data2.rename(columns = {"winner_id" : "loser_id", "loser_id" : "winner_id"})
        data2["target"] = 0
        data2 = data2[["winner_id", "loser_id", "Date", "target", "surface", "tourney_id"]]
        
        tot = pd.concat([data1, data2], axis= 0)
        tot["Date"] = pd.to_datetime(tot["Date"], format = "%Y-%m-%d")
        
        correlation_surface   = calculate_corr_surface(tot, redo)
        correlation_time      = calculate_corr_time(tot, redo)
        
    else:    
        correlation_surface   = calculate_corr_surface(data, redo)
        correlation_time      = calculate_corr_time(data, redo)
        
    data["ref_days"]= (data["Date"]- pd.to_datetime("01/01/1901")).dt.days
    data["diff_fatigue_games"] = data[["ref_days", "winner_id", "loser_id", "total_games"]].apply(lambda x : fatigue_games(x, data), axis= 1)
    del data["ref_days"]
    
    t0 = time.time()
    col_for_stats = ['Date', 'winner_id', 'loser_id', "surface", 'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
                     'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced','w_1st_srv_ret_won',
                     'w_2nd_srv_ret_won', 'w_bp_converted', 'w_total_srv_won', 'w_total_ret_won', 'l_1st_srv_ret_won', 'l_2nd_srv_ret_won', 'l_bp_converted',
                     'l_total_srv_won', 'l_total_ret_won', 'w_tie-breaks_won', 'l_tie-breaks_won', 'Nbr_tie-breaks', "N_set"]
    
#    counts = parallelize_dataframe(data[["Date", "winner_id", "loser_id", "surface"]], execute_stats, [data[col_for_stats], correlation_surface, correlation_time], 8)
    counts = data[["Date", "winner_id", "loser_id", "surface"]].apply(lambda x : weighted_statistics(x, [data[col_for_stats], correlation_surface, correlation_time]), axis= 1)
    counts = list(zip(*counts["surface"]))
    
    stats_cols = ["Common_matches", "diff_aces", "diff_df", "diff_1st_serv_in", "diff_1st_serv_won", "diff_2nd_serv_won",
                             "diff_skill_serv", "diff_skill_ret", "diff_overall_skill", "diff_serv1_ret2", "diff_serv2_ret1", "diff_bp", "diff_tie_break",
                             "diff_victories_12", "diff_victories_common_matches"]
    for i, col in enumerate(stats_cols):
        data[col] =  list(counts[i])

    data = global_stats(data)
    print("exec stats {}".format(time.time()-t0))
    
    data["target"] = 1
    basic_cols    = ["target", "Date", "winner_id", "loser_id", "tourney_name", 'prize', 'best_of', 'round', 'Common_matches']
    
    stats_cols =  ['diff_aces', 'diff_df', 'diff_1st_serv_in', "diff_fatigue_games",
                      'diff_1st_serv_won','diff_2nd_serv_won', 'diff_skill_serv','diff_skill_ret', 'diff_overall_skill', 'diff_serv1_ret2','diff_serv2_ret1', 
                      'diff_bp', 'diff_victories_12', 'diff_victories_common_matches','diff_age', 'diff_ht', "diff_days_since_stop",
                      'diff_weight', 'diff_year_turned_pro', 'diff_elo', 'diff_rank', 'diff_rk_pts', 'diff_hand', 'diff_is_birthday', 'diff_home']
                            
    data2 = data.copy()
    data2["target"] = 0
    for col in stats_cols:
        data2[col] = -1*data2[col]
        
    total_data_with_cols = pd.concat([data, data2], axis= 0)
    total_data           = pd.concat([data[stats_cols + basic_cols], data2[stats_cols + basic_cols]], axis= 0)
                    
    return total_data_with_cols, total_data


if __name__ == "__main__": 
    import os
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"

    data = pd.read_csv(r"C:\Users\User\Documents\tennis\data\clean_datasets\historical\matches_elo_variables_V1.csv")
    data["Date"] = pd.to_datetime(data["Date"], format = "%Y-%m-%d")
    data["DOB_w"] = pd.to_datetime(data["DOB_w"], format = "%Y-%m-%d")
    data["DOB_l"] = pd.to_datetime(data["DOB_l"], format = "%Y-%m-%d")
    tot = create_statistics(data)
    
    tot2 = tot[['Common_matches', 'diff_aces', 'diff_df', 'diff_1st_serv_in', 'diff_1st_serv_won','diff_2nd_serv_won', 'diff_skill_serv',
                 'diff_skill_ret', 'diff_overall_skill', 'diff_serv1_ret2','diff_serv2_ret1', 'diff_bp', 'diff_victories_12', 'diff_victories_common_matches',
                 'diff_age', 'diff_ht', 'diff_weight', 'diff_year_turned_pro', 'diff_elo', 'diff_rank', 'diff_rk_pts', 'diff_hand', 'diff_is_birthday', 'diff_home']]
    tot2 = tot2.loc[tot2["Common_matches"] > 5]