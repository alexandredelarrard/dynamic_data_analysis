# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:16:28 2018

@author: JARD
"""

import pandas as pd
import os
import warnings
import time
warnings.filterwarnings("ignore")

from create_data.data_creation.extract_data_atp import import_data_atp
from create_data.data_creation.create_statistics_historyV2 import create_statistics
from create_data.data_update.update_data import update_stable
from create_data.data_creation.create_elo_rankingV2 import merge_data_elo
from create_data.data_creation.create_variables import prep_data

os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"

def main_create_data(param):
    
    rebuild = True in list(param.values())
    
    if rebuild:
            
        if param["redo_missing_atp_statistics"]: #// 10 min 
            ### read atp data and clean it / redo = build from scratch with the matching algo with stats match from atp 
            path = os.environ["DATA_PATH"]  + "/brute_info/historical/brute_info_atp/"
            data_atp = import_data_atp(path, redo = False) ### redo the stats match with crawled matches from atp
        else:
            data_atp = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_V1.csv")
            data_atp["Date"]= pd.to_datetime(data_atp["Date"], format = "%Y-%m-%d")

        if param["create_elo"]:
            ### add elo system ranking # // 20 min 
            data_merge_player_elo = merge_data_elo(data_atp)
            data_merge_player_elo.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_V1.csv", index= False)
        
        else:
            data_merge_player_elo = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_V1.csv")
            data_merge_player_elo["Date"]= pd.to_datetime(data_merge_player_elo["Date"], format = "%Y-%m-%d")
            data_merge_player_elo["DOB_w"] = pd.to_datetime(data_merge_player_elo["DOB_w"], format = "%Y-%m-%d")
            data_merge_player_elo["DOB_l"] = pd.to_datetime(data_merge_player_elo["DOB_l"], format = "%Y-%m-%d")
            
        ### create value added variables/lean dataset and irregularities
        if param["create_variable"]: #// 0.5 min 
            data2 = prep_data(data_merge_player_elo, verbose=0)
            data2.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_variables_V1.csv", index= False)
        else:
            data2 = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_variables_V1.csv")
            data2["Date"]= pd.to_datetime(data2["Date"], format = "%Y-%m-%d")
            data2["DOB_w"] = pd.to_datetime(data2["DOB_w"], format = "%Y-%m-%d")
            data2["DOB_l"] = pd.to_datetime(data2["DOB_l"], format = "%Y-%m-%d")
        
        ### create counting past historical data
        if param["create_statistics"]: #// 200 min 
            data_total = create_statistics(data2, redo= False)
             
            #### save dataset
            data_total.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/overall/stable/hictorical_origin/total_dataset_modelling.csv", index= False)
            
        if param["update_data"]:
            data_total = update_stable()
           
    if not rebuild:        
        data_total = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/overall/updated/latest/total_dataset_modelling.csv")
        data_total["Date"] = pd.to_datetime(data_total["Date"], format = "%Y-%m-%d")
            
    return data_total
    

def main_creation(rebuild=False):
    
    t0 = time.time()
    if rebuild == False:
        rebuild = {"redo_missing_atp_statistics" : False,
                   "create_elo" : False,
                   "create_statistics": False,
                   "create_variable" : False,
                   "update_data" : False}
        
    if rebuild == True:
         rebuild = {"redo_missing_atp_statistics" : True,
                   "create_elo" : True,
                   "create_statistics": True,
                   "create_variable" : True,
                   "update_data":True}    
   
    full_data = main_create_data(rebuild)    
    print("\n \n Global time to create data is {0}".format(time.time() - t0))
    
    return full_data


if __name__ == "__main__":
    rebuild = {"redo_missing_atp_statistics" : False,
                   "create_elo" : False,
                   "create_statistics": False,
                   "create_variable" : False,
                   "update_data":True}    
     
    full_data = main_creation(rebuild=rebuild)
    
    
    