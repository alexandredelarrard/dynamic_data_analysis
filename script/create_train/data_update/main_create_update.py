# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 14:23:29 2018

@author: User
"""

import sys
import os
import pandas as pd
import glob

sys.path.append(r"C:\Users\User\Documents\tennis")
from crawling.crawling_tourney import updated_tourney
from crawling.crawling_atp_ranking import atp_crawl
from crawling.crawling_atp_players import updated_players

from create_train.data_update.update_data import update_stable, clean_new_matches

def create_update(boolean_update):
    
    #### import history and append extractions after, save to latest
    print("\n from stable/historical")
    path = os.environ["DATA_PATH"] + "/clean_datasets/overall/stable/hictorical_origin/total_dataset_modelling.csv"
    latest_data = pd.read_csv(path)
    
    path =os.environ["DATA_PATH"] + "/clean_datasets/overall/stable/all_extractions"
    files_already_there = glob.glob(path + "/*.csv")
    for f in files_already_there:
        latest_data = pd.concat([latest_data, pd.read_csv(f)], axis =0)
        print("{0} new shape train = {1}".format(f.replace(path,"").replace(".csv",""),latest_data.shape))
    latest_data = latest_data.reset_index(drop=True)    
    
    if boolean_update:
        #### check all ranks and tourney are downloaded
        print("[0-0] check tourney")
        updated_tourney()
        
        #### check all ranks and tourney are downloaded
        print("[0-1] check ranks")
        atp_crawl()
        
        #### check all recent matches into data
        print("[0-2] check all recent matches downloaded")
        update_stable(latest_data)
        
        #### check all players into data
        print("[0-3] check all players in dataset")
        updated_players()
        
        #### check all players into data
        print("[0-4] aggregate new data")
        clean_new_matches(latest_data)
    
    return latest_data


if __name__ =="__main__":
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"
    matches = create_update(boolean_update= True)
    