# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:16:28 2018

@author: JARD
"""

import pandas as pd
import os
from modelling.modelling_lr import modelling_logistic
from data_prep.extract_data_atp import import_data_atp
#from data_prep.create_statistics_history import data_prep_history
from data_prep.create_elo_ranking import merge_data_elo
from data_prep.create_variables import prep_data

os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"

def main_create_data(rebuild):
    
    if rebuild:
        ### read atp data and clean it / redo = build from scratch with the matching algo with stats match from atp 
        path = os.environ["DATA_PATH"]  + "/brute_info/historical/brute_info_atp/"
        data_atp = import_data_atp(path, redo = False)

        ### add elo system ranking
        data_merge_player_elo = merge_data_elo(data_atp)
        
        ### create value added variables/lean dataset and irregularities
        data2 = prep_data(data_merge_player_elo)
   
        ### create counting past historical data
         
        #### save dataset
        data2.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_variables_V1.csv", index= False)
        
        
    else:
        data2 = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_variables_V1.csv")
        data2["Date"] = pd.to_datetime(data2["Date"], format = "%Y-%m-%d")
        data2["DOB_w"] = pd.to_datetime(data2["DOB_l"], format = "%Y-%m-%d")
        data2["DOB_l"] = pd.to_datetime(data2["DOB_l"], format = "%Y-%m-%d")
        data2["tourney_date"] = pd.to_datetime(data2["tourney_date"], format = "%d/%m/%Y")
        
    return data2
    

if __name__ == "__main__":

    data_atp = main_create_data(rebuild= True)
    