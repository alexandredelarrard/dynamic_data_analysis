# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:38:37 2018

@author: User
"""

import pandas as pd
import os

from extract_additionnal_data import extraction_atp


def prepare_extraction(data_extract):
    return data

def update_stable():
    
    path = os.environ["DATA_PATH"] + "/clean_datasets/overall/stable/total_dataset_modelling.csv"
    latest_data = pd.read_csv(path)
    latest_data = latest_data.loc[latest_data["target"] == 1]
    latest_data = latest_data.sort_values(["tourney_date_x", "tourney_name"])
    
    latest = {"Date": latest_data["tourney_date_x"].max(), 
              "tourney_name": latest_data.loc[latest_data["tourney_date_x"] == latest_data["tourney_date_x"].max(), "tourney_name"].tolist()[-1],
              }
    
    return latest, latest_data

if __name__ == "__main__":
    
    latest, latest_data = update_stable()
    