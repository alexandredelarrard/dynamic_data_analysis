# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:16:28 2018

@author: JARD
"""

import pandas as pd
import os
from modelling.modelling_lr import modelling_logistic
from data_prep.extract_data_atp import import_data_atp
from data_prep.create_statistics_history import data_prep_history
from data_prep.create_elo_ranking import merge_data_elo

os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"

def main_create_data(rebuild):
    
    ### read atp data and clean it / redo = if redo the matching algo with stats match
    if rebuild:
        data_atp = import_data_atp(os.environ["DATA_PATH"]  + "/brute_info/historical/brute_info_atp/", redo = False)
        data_atp.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_V1.csv", index= False)
        
    else:
        data_atp = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_V1.csv")
        
    ### add elo system ranking
    data_merge_player_elo = merge_data_elo(data_atp)
    
    ### create value added variables
    data2 = data_prep_history(data_merge_player_elo)
    
    return data_merge_player_elo
    

if __name__ == "__main__":
    data_atp = main_create_data(rebuild= False)
    
#    #### lr modelling 
#    clf, importance = modelling_logistic(data_merge_player_elo, "2017-01-01", "2017-06-01", "gbm")