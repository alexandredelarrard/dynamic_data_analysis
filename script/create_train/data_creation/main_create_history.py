# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:16:28 2018

@author: JARD
"""
import os
import pandas as pd

os.chdir(r"C:\Users\User\Documents\tennis\dynamic_data_analysis\script")
from create_train.data_creation.extract_data_atp import import_data_atp
from create_train.data_creation.create_statistics_historyV2 import create_statistics
from create_train.data_creation.create_elo_rankingV2 import merge_data_elo
from create_train.data_creation.create_variables import prep_data

def create_history(rebuild):

    if rebuild:
          
        try:
            os.remove(os.environ["DATA_PATH"]  + "/clean_datasets/overall/stable/hictorical_origin/total_dataset_modelling.csv")
            os.remove(os.environ["DATA_PATH"]  + "/clean_datasets/overall/updated/latest/total_dataset_modelling.csv")
        except Exception:
            pass
        
        ### read atp data and clean it / redo = build from scratch with the matching algo with stats match from atp 
        path = os.environ["DATA_PATH"]  + "/brute_info/historical/brute_info_atp/"
        data_atp = import_data_atp(path, redo = False) ### redo the stats match with crawled matches from atp
       
        ### add elo system ranking # // 20 min 
        data_merge_player_elo = merge_data_elo(data_atp)
        data_merge_player_elo.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_V1.csv", index= False)
        
        ### create value added variables/lean dataset and irregularities
        #// 0.5 min 
        data2 = prep_data(data_merge_player_elo, verbose=0)
        data2.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_variables_V1.csv", index= False)
        
        ### create counting past historical data
        #// 250 min 
        data_total = create_statistics(data2, redo= False)
         
        #### save dataset
        data_total.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/overall/stable/hictorical_origin/total_dataset_modelling.csv", index= False)
       
    if not rebuild: 
        try:
            data_total = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/overall/updated/latest/total_dataset_modelling.csv")
            data_total["Date"] = pd.to_datetime(data_total["Date"], format = "%Y-%m-%d")
            
            ### create a report on the original dataset
            print(data_total.shape)
            
        except Exception:
            print(" You need to create the historical data with rebuild = True before assessing it")
            pass
            

if __name__ == "__main__":
    full_data = create_history(rebuild=True)
    
    
    