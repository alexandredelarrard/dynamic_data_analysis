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

def import_data():
    
    #### import history and append extractions after, save to latest
    print(" ----- from stable/historical")
    path = os.environ["DATA_PATH"] + "/clean_datasets/overall/stable/hictorical_origin/total_dataset_modelling.csv"
    latest_data = pd.read_csv(path)
    
    path =os.environ["DATA_PATH"] + "/clean_datasets/overall/stable/all_extractions"
    files_already_there = glob.glob(path + "/*.csv")
    for f in files_already_there:
        latest_data = pd.concat([latest_data, pd.read_csv(f)], axis =0)
        print(" ----- {0} new shape train = {1}".format(f.replace(path,"").replace(".csv",""),latest_data.shape))
       
    latest_data = latest_data.reset_index(drop=True)   
    latest_data["Date"] = pd.to_datetime(latest_data["Date"], format = "%Y-%m-%d")
    latest_data["tourney_date"] = pd.to_datetime(latest_data["tourney_date"], format = "%Y-%m-%d")
    latest_data["DOB_w"] = pd.to_datetime(latest_data["DOB_w"], format = "%Y-%m-%d")
    latest_data["DOB_l"] = pd.to_datetime(latest_data["DOB_l"], format = "%Y-%m-%d")
    
    return latest_data


def create_update(boolean_update):
    
    if boolean_update:
        latest_data = import_data()
        
        #### check all ranks and tourney are downloaded
        print("[2-0] check tourney")
        updated_tourney()
        
        #### check all ranks and tourney are downloaded
        print("[2-1] check ranks")
        atp_crawl()
        
        #### check all recent matches into data
        print("[2-2] check all recent matches downloaded")
        update_stable(latest_data)
        
        #### check all players into data
        print("[2-3] check all players in dataset")
        latest_matches = pd.read_csv(os.environ["DATA_PATH"] + "/clean_datasets/overall/updated/extracted/extraction_brute.csv")
        updated_players(latest_matches)
        
        #### check all players into data
        print("[2-4] aggregate new data")
        clean_new_matches(latest_data)
        

if __name__ =="__main__":
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"
    matches = create_update(boolean_update= True)
    