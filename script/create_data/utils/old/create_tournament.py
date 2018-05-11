# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:14:40 2018

@author: JARD
"""

import pandas as pd
from datetime import timedelta

import time
import os



def merge_with_tournois(data_concat):
    
    
    t0 = time.time()
    tournois = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/tournament/tournaments.csv", encoding = "latin1")[["Tournament", 'Prize', 'Surface', 'Indoor_flag',
       'Currency', 'ID', 'pays', 'City', "Date_start_tournament"]]
    tournois["Date_start_tournament"] = pd.to_datetime(tournois["Date_start_tournament"], format = "%d/%m/%Y")
    tournois["year"] = (tournois["Date_start_tournament"] +  timedelta(days=7)).dt.year
    tournois["key"] = tournois["Tournament"].astype(str) + "-" + tournois["year"].astype(str) 
    
    data_concat["Tournament"] = data_concat["Tournament"].apply(lambda x: x.replace("’", " "))
    data_concat["year"] = (data_concat["Date"] +  timedelta(days=7)).dt.year
    data_concat["key"] = data_concat["Tournament"].astype(str) + "-" + data_concat["year"].astype(str) 
    
    data_tournois = pd.merge(data_concat, tournois, on = ["key"], how = "left")
    data_tournois.drop(["Surface_y", "key", "year_x", "year_y", "Tournament_y", "Indoor_flag"], axis= 1, inplace = True)
    
    print("[{0}s] 2) Merge origin data with tournement data ".format(time.time() - t0))
    
    return data_tournois


    