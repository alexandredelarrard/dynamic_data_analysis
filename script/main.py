# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:16:28 2018

@author: JARD
"""

import pandas as pd
import os

from data_prep.extract_data_atp import import_data_atp
from data_prep.create_statistics_history import create_statistics
from data_prep.create_elo_rankingV2 import merge_data_elo
from data_prep.create_variables import prep_data

os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"

def main_create_data(param):
    
    rebuild = True in list(param.values())
    
    if rebuild:
        if param["redo_missing_atp_statistics"]:
            ### read atp data and clean it / redo = build from scratch with the matching algo with stats match from atp 
            path = os.environ["DATA_PATH"]  + "/brute_info/historical/brute_info_atp/"
            data_atp = import_data_atp(path, redo = False)
        else:
            data_atp = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_V1.csv")
            data_atp["Date"]= pd.to_datetime(data_atp["Date"], format = "%Y-%m-%d")

        if param["create_elo"]:
            ### add elo system ranking
            data_merge_player_elo = merge_data_elo(data_atp)
            data_merge_player_elo.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_V1.csv", index= False)
        
        else:
            data_merge_player_elo = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_V1.csv")
            data_merge_player_elo["Date"]= pd.to_datetime(data_merge_player_elo["Date"], format = "%Y-%m-%d")
            
        ### create value added variables/lean dataset and irregularities
        if param["create_variable"]:
            data2 = prep_data(data_merge_player_elo)
            data2.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_variables_V1.csv", index= False)
        else:
            data2 = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_variables_V1.csv")
            data2["Date"]= pd.to_datetime(data2["Date"], format = "%Y-%m-%d")
        
        ### create counting past historical data
        if param["create_statistics"]:
            data_total, data3 = create_statistics(data2, redo= False)
             
            #### save dataset
            data3.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/variables_for_modelling_V1.csv", index= False)
            data_total.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/total_dataset_modelling.csv", index= False)
            
    if not rebuild:        
        data3 = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/variables_for_modelling_V1.csv")
        data3["Date"] = pd.to_datetime(data3["Date"], format = "%Y-%m-%d")
        
        data_total = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/total_dataset_modelling.csv")
        data_total["Date"] = pd.to_datetime(data3["Date"], format = "%Y-%m-%d")
            
    return data_total, data3
    

if __name__ == "__main__":
    rebuild = {
               "redo_missing_atp_statistics" : False,
               "import_atp": False,
               "create_elo": False,
               "create_variable" : False,
               "create_statistics" : True}
    
    data_atp = main_create_data(rebuild)
#    data2 = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_variables_V1.csv")
#    
#    data0 = data_atp[1].copy()
#    data0 = data0.loc[~pd.isnull(data0["diff_aces"])]
#    