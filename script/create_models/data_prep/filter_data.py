# -*- coding: utf-8 -*-
"""
Created on Sun May 13 09:02:29 2018

@author: User
"""

import pandas as pd
import numpy as np

def data_prep_for_modelling(data):
    
    full_data = data.copy()
    
    shape0 = full_data.shape[0]
    full_data = full_data.loc[~pd.isnull(full_data["diff_aces"])&(full_data["Common_matches"]>5)]
    print("[0] Suppressed missing values : {} suppressed".format(full_data.shape[0] - shape0))
    
    shape0 = full_data.shape[0]
    full_data = full_data.loc[full_data['missing_stats'] !=1]
    print("[0] Suppressed missing_stats = 1 : {} suppressed".format(full_data.shape[0] - shape0))

    ### filter interesting columns
    columns_to_keep = list(set(['diff_fatigue_games', 'Common_matches', 'diff_aces', 'diff_df', 'diff_1st_serv_in', 'diff_1st_serv_won', 'diff_2nd_serv_won', 
                     'diff_skill_serv', 'diff_skill_ret', 'indoor_flag',  'week', 'best_of', 'match_num', # 'prize',  'diff_days_since_stop', 'prob_elo',  'diff_hand', 'diff_is_birthday',
                     'diff_overall_skill', 'diff_serv1_ret2', 'diff_serv2_ret1', 'diff_bp', 'diff_tie_break', 'diff_victories_12', 'diff_victories_common_matches', 
                     'diff_pts_common_matches', 'diff_mean_rank_adversaries', 'diff_age', 'diff_ht', 'diff_imc', 'diff_weight', 'diff_year_turned_pro', 'diff_elo',
                     'diff_rank', 'diff_rk_pts', 'diff_home', 'max_rank', 'target', 'tourney_level', 'masters', 'Currency', 'round']))
   
    full_data2 = full_data[columns_to_keep].copy()
    
    full_data2["RR_round"] = np.where(full_data2['round'] == "RR", 1,0)
    
    ### take care of round
    dico = {"R32": 32, "R16": 16, "R64": 64, "R128":128, "QF": 8, "SF": 4, "F":2, "RR": 4 }
    full_data2['round'] = full_data2['round'].map(dico).astype(int)  
    
    for col in ['tourney_level', 'masters', 'Currency']:
        a = pd.get_dummies(full_data[col], prefix = col)
        full_data2 = pd.concat([full_data2, a], axis=1)
        del full_data2[col]
        
    ### keep for training filtering
    full_data2["Date"] = full_data["Date"]
    full_data2["tourney_name"] = full_data["tourney_name"]
    
    full_data2["prob_elo"] =  1 / (1 + 10 ** ((full_data2["diff_elo"]) / 400))
    
   
    #
    return full_data2


def data_analysis(full_data):
    
    ### people with fewer service point than the minimum with game numbers
    full_data.loc[full_data["w_svpt"] < full_data["total_games"]*4/2].drop_duplicates().to_csv(r"C:\Users\User\Documents\tennis\data\to_check\winner_service_weird.csv")
    