# -*- coding: utf-8 -*-
"""
Created on Sun May 13 09:02:29 2018

@author: User
"""

import pandas as pd
import numpy as np
import os
#from matplotlib import pyplot as plt
#from create_train.utils.plot_lib import var_vs_target

def data_prep_for_modelling(full_data, start_year= 1991):
    """
    filter data for modelling 
    - 156300, 144 variables
    """
    
    shape0 = full_data.shape[0]
    print("Number of matches under 5 common adversaries played is {0}".format(full_data.loc[full_data["Common_matches"]<30].shape[0]))
    
    full_data = full_data.loc[~pd.isnull(full_data["diff_aces"])&(full_data["Common_matches"]>=30)&(full_data["Date"].dt.year>= start_year)]
    
    try:
        full_data = full_data.loc[~pd.isnull(full_data["l_2nd_srv_ret_won"])]
        full_data = full_data.loc[~pd.isnull(full_data["w_2nd_srv_ret_won"])]
        full_data["total_tie_break_l"] =  full_data["total_tie_break_l"].fillna(-1)
        full_data["total_tie_break_w"] =  full_data["total_tie_break_w"].fillna(-1)
    except Exception:
        pass
    full_data = full_data.loc[~pd.isnull(full_data["diff_overall_skill"])]
    print("[0] Suppressed missing values : {} suppressed".format(full_data.shape[0] - shape0))
    
    # =============================================================================
    #     ###fill na to -1 or delete observations
    # =============================================================================
    full_data["diff_serv2_ret1"]   =  full_data["diff_serv2_ret1"].fillna(-1)
    full_data["diff_skill_ret"]    =  full_data["diff_skill_ret"].fillna(-1)
    full_data["prop_victory_surface_w"]  =  full_data["prop_victory_surface_w"].fillna(-1)
    full_data["prop_victory_surface_l"]  =  full_data["prop_victory_surface_l"].fillna(-1)
    full_data["prop_last_set_gagne_l"]   =  full_data["prop_last_set_gagne_l"].fillna(-1)
    full_data["prop_last_set_gagne_w"]   =  full_data["prop_last_set_gagne_w"].fillna(-1)
    full_data['winner_seed'] = full_data['winner_seed'].fillna(-1)
    full_data['loser_seed'] = full_data['winner_seed'].fillna(-1)
    
    ### last minute feature engineering
    full_data["diff_seed"] = full_data['winner_seed']  - full_data['loser_seed'] 
    
    full_data['winner_entry'] = full_data['winner_entry'].fillna("Normal")
    full_data['loser_entry'] = full_data['loser_entry'].fillna("Normal")
    full_data['winner_entry'] = np.where(full_data['winner_entry'] == "Normal", 3,
                                np.where(full_data['winner_entry'] == "Q", 2, 13))
    full_data['loser_entry'] =  np.where(full_data['loser_entry']== "Normal", 3,
                                np.where(full_data['loser_entry'] == "Q", 2, 1))
    full_data['diff_entry'] = full_data['winner_entry'].astype(int) - full_data['loser_entry'].astype(int)
    
    full_data['masters'] = np.where(full_data['masters'] == "grandslam",1,0)
    
    # =============================================================================
    #     ### filter interesting columns
    # =============================================================================
    columns_to_keep = ["tourney_name", "Date", 'target',  ### for filter or output
                      'Common_matches', 'bst_rk_l', 'bst_rk_w',  'day_of_year', 'diff_1st_serv_in',
                     'diff_1st_serv_won', 'diff_2nd_serv_won', 'diff_aces', "winner_age", "loser_age", 'diff_bp', 'diff_df',  'diff_hand',
                     'diff_home', "winner_ht", "loser_ht", 'diff_imc', 'diff_mean_rank_adversaries', 'diff_overall_skill', 'diff_pts_common_matches', 'diff_rank', 'diff_rk_pts',
                     'diff_serv1_ret2', 'diff_serv2_ret1', 'diff_skill_ret', 'diff_skill_serv', 'diff_tie_break',  'diff_victories_common_matches',
                     'diff_weights', 'diff_year_turned_pro', 'draw_size', 'round', 'nbr_reach_level_tourney_l','diff_victories_12',
                     'nbr_reach_level_tourney_w', 'prize', 'prop_last_set_gagne_l', 'prop_last_set_gagne_w', 'prop_victory_surface_l', 'prop_victory_surface_w',
                     'surface', 'loser_rank_points', 'winner_rank_points', 
                     'diff_entry','diff_fatigue_minutes','masters','diff_weight', 'diff_elo',
                     'diff_win_set_1', 'diff_win_set_2', 'diff_time_on_court','diff_time_set','diff_match_start_season','diff_weak_hand',
                     
                     ####   suppress 'indoor_flag', 'best_of','diff_seed', 'prob_elo','day_week','diff_age',
                     ####  'diff_fatigue_games', 'w_birthday', 'l_birthday','winner_rank','loser_rank', 'diff_ht', 
                     ]
   
    full_data = full_data[columns_to_keep]
    
    # =============================================================================
    #     ### take care of object variables
    # =============================================================================
    for col in ["surface"]:
        a = pd.get_dummies(full_data[col], prefix = col)
        full_data = pd.concat([full_data, a], axis=1)
        del full_data[col]
        
    full_data = full_data.drop(["surface_Hard_0"],axis=1)
        
    return full_data


if __name__ == "__main__": 
    
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"
#    full = data_prep_for_modelling()
