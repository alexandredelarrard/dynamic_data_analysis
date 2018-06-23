# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:16:28 2018

@author: JARD
"""
import os

os.chdir(r"C:\Users\User\Documents\tennis\dynamic_data_analysis\script")
from create_train.data_creation.extract_data_atp import import_data_atp
from create_train.data_creation.create_statistics_historyV2 import create_statistics
from create_train.data_creation.create_elo_rankingV2 import merge_data_elo
from create_train.data_creation.create_variables import prep_data

def create_history(rebuild):

    if rebuild:
        redo = False  #### will re extract to complete missing values and correlation matrix
        
        try:
            os.remove(os.environ["DATA_PATH"]  + "/clean_datasets/overall/stable/hictorical_origin/total_dataset_modelling.csv")
        except Exception:
            pass
        
        # =============================================================================
        #         ### read atp data and clean it / redo = build from scratch with the matching algo with stats match from atp 
        # =============================================================================
        
        path = os.environ["DATA_PATH"]  + "/brute_info/historical/brute_info_atp/"
        data_atp = import_data_atp(path, redo = redo) ### redo the stats match with crawled matches from atp
       
        # =============================================================================
        #         ### add elo system ranking # // 20 min 
        # =============================================================================
        
        data_merge_player_elo = merge_data_elo(data_atp)
        data_merge_player_elo.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_V1.csv", index= False)
        
        # =============================================================================
        #         #data_merge_player_elo = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_V1.csv")
        #         ### create value added variables/lean dataset and irregularities // 0.5 min 
        # =============================================================================
        
        data2 = prep_data(data_merge_player_elo, verbose=0)
        data2.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_variables_V1.csv", index= False)
        
        # =============================================================================
        #         #data2 = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_variables_V1.csv")
        #         ### create counting past historical data // 90 min 
        # =============================================================================
        
        data_total = create_statistics(data2, redo= redo)
        data_total.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/overall/stable/hictorical_origin/total_dataset_modelling.csv", index= False)
       
    if not rebuild: 
        if not os.path.isfile(os.environ["DATA_PATH"]  + "/clean_datasets/overall/stable/hictorical_origin/total_dataset_modelling.csv"):
            print(" You need to create the historical data. Switch to rebuild = True. It will take few hours to run so take a nap")
            create_history(rebuild=True)
            
            
if __name__ == "__main__":
    full_data = create_history(rebuild=True)
    
    
    