# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:38:37 2018

@author: User
"""

import pandas as pd
import os
import sys
import time

from data_update.clean_updated_data import clean_extract
from create_data.data_creation.create_statistics_historyV2 import *

sys.path.append(r"C:\Users\User\Documents\tennis")
from crawling.crawling_additionnal_data import extract_additionnal_data
from crawling.crawling_atp_ranking import atp_crawl

def update_stable():
    
    path = os.environ["DATA_PATH"] + "/clean_datasets/overall/stable/total_dataset_modelling.csv"
    latest_data = pd.read_csv(path)
    latest_data = latest_data.loc[latest_data["target"] == 1]
    latest_data = latest_data.sort_values(["tourney_date_x", "tourney_name"])
    
    latest = {"Date": latest_data["tourney_date_x"].max()}
    latest= {"Date": "2018-05-21"}
    
    ### crawl rank
    t0 = time.time()
    atp_crawl(latest)
    print("time for atp rank crawling {0}".format(time.time() - t0))

    ### crawl data   
    t0 = time.time()
    extract_additionnal_data(latest)
    print("time for atp latest games crawling {0}".format(time.time() - t0))
    
    ### clean the crawled data
    extra = clean_extract()
    
    ### calculate elo
    
    
    ### calculate the statistics on it
    
    
    return extra


if __name__ == "__main__":
    extra = update_stable()
    