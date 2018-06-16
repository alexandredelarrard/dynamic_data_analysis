# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 14:23:29 2018

@author: User
"""

import sys
import os
import pandas as pd

sys.path.append(r"C:\Users\User\Documents\tennis")
from crawling.crawling_tourney import updated_tourney
from crawling.crawling_atp_ranking import atp_crawl

#from update_data import update_stable

def create_update(boolean_update):
    
    #### import history and append extractions after, save to latest
    print("from stable/historical")
    path = os.environ["DATA_PATH"] + "/clean_datasets/overall/stable/hictorical_origin/total_dataset_modelling.csv"
    latest_data = pd.read_csv(path)
 
    if boolean_update:
        #### check all ranks and tourney are downloaded
        print("[0-0] check tourney")
        updated_tourney()
        
        #### check all ranks and tourney are downloaded
        print("[0-1] check ranks")
        atp_crawl()
        
        #### check all recent matches into data
        print("[0-2] check all recent matches downloaded")
#        new_data_modelling = update_stable(latest_data)
        
        #### check all players into data
        print("[0-3] check all players in dataset")
    #    updated_tourney()

    
    return latest_data

if __name__ =="__main__":
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"
    matches = create_update()
    