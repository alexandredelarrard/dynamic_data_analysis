# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:38:37 2018

@author: User
"""

import pandas as pd
import os
import sys

from clean_updated_data import clean_extract

sys.path.append("C:\Users\User\Documents\tennis")
from crawling.crawling_additionnal_data import extract_additionnal_data


def update_stable():
    
    path = os.environ["DATA_PATH"] + "/clean_datasets/overall/stable/total_dataset_modelling.csv"
    latest_data = pd.read_csv(path)
    latest_data = latest_data.loc[latest_data["target"] == 1]
    latest_data = latest_data.sort_values(["tourney_date", "tourney_name"])
    
    latest = {"Date": latest_data["tourney_date"].max(), 
              "tourney_name": latest_data.loc[latest_data["tourney_date"] == latest_data["tourney_date"].max(), "tourney_name"].tolist()[-1]}

    ### crawl data    
    extra = extract_additionnal_data(latest)
    
    ### clean the crawled data
    extra = clean_extract(extra)
    
    return extra

if __name__ == "__main__":
    
    latest, latest_data = update_stable()
    