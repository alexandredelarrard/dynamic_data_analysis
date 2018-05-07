# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:16:28 2018

@author: JARD
"""

import pandas as pd
import os

from data_prep.extract_data_atp import import_data_atp
from data_prep.create_statistics_history import create_statistics
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
        data_merge_player_elo.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_V1.csv", index= False)
        
        ### create value added variables/lean dataset and irregularities
        data2 = prep_data(data_merge_player_elo)
        data2.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_variables_V1.csv", index= False)
   
        ### create counting past historical data
        data_total, data3 = create_statistics(data2, redo= False)
         
        #### save dataset
        data3.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/variables_for_modelling_V1.csv", index= False)
        data_total.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/total_dataset_modelling.csv", index= False)
        
    else:
        data3 = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/variables_for_modelling_V1.csv")
        data3["Date"] = pd.to_datetime(data3["Date"], format = "%Y-%m-%d")
        
        data_total = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/total_dataset_modelling.csv")
        data_total["Date"] = pd.to_datetime(data3["Date"], format = "%Y-%m-%d")
        
    return data_total, data3
    

if __name__ == "__main__":

    data_atp = main_create_data(rebuild= True)
#    data2 = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_variables_V1.csv")
    