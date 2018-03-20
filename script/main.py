# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:16:28 2018

@author: JARD
"""

import pandas as pd
import os
from modelling.modelling_lr import modelling_logistic
from data_prep.create_tournament import merge_with_tournois
from data_prep.extract_data_origin import import_data_origin 
from data_prep.extract_data_atp import import_data_atp, merge_match_ID, merge_origin_atp
from data_prep.create_statistics_history import data_prep_history
from data_prep.create_elo_ranking import merge_data_elo
from utils.plot_lib import var_vs_target
from utils.create_id_player import players_ID_creation


os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"


if __name__ == "__main__":
    
    ### read data and clean it 
    data_origin = import_data_origin(os.environ["DATA_PATH"]  + "/brute_info/historical/brute_info_origin/")
    
    ### extract statistics and create target variable
    data_origin_tournament = merge_with_tournois(data_origin, os.environ["DATA_PATH"]  + "/clean_datasets/tournament/")
    
    data_origin_tournament_ID = merge_match_ID(data_origin_tournament, key ="ORIGIN_ID")
    
    ### read atp data and clean it
    data_atp = import_data_atp(os.environ["DATA_PATH"]  + "/brute_info/historical/brute_info_atp/")
    
    #### merge atp origin
    data_merge = merge_origin_atp(data_origin_tournament_ID, data_atp, common_key = "ATP_ID")
    
#    missing_values = data_merge.loc[pd.isnull(data_merge["w_ace"]), ["winner_name", "loser_name", "tourney_name", "Date"]]
#    missing_values["year"] = missing_values["Date"].dt.year
#    missing_values["keys"] = missing_values["year"].astype(str) + " "+missing_values["tourney_name"].str.lower() + " "+missing_values["winner_name"] +" "+ missing_values["loser_name"]
#    
#    total_data["keys"]
    
    #### create_ID_players
#    players_ID_creation(data_merge)
    
    ### add elo system ranking
#    data_elo = merge_data_elo(data)
    
    #data2 = data_prep_history(data_origin)

#    #### lr modelling 
#    clf, importance = modelling_logistic(data_elo, "2017-01-01", "2017-06-01", "gbm")
 