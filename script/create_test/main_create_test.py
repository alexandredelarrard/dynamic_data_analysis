# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 14:20:34 2018

@author: User
"""

import os
import sys
from datetime import datetime
import pandas as pd

os.chdir(r"C:\Users\User\Documents\tennis\dynamic_data_analysis\script")
sys.path.append(r"C:\Users\User\Documents\tennis\dynamic_data_analysis\script")
from create_test.data_creation.clean_test_data import clean, add_elo, calculate_stats
from create_train.data_update.main_create_update import create_update, import_data

sys.path.append(r"C:\Users\User\Documents\tennis\crawling")
from crawling_futur_match import extract_futur_data
from crawling_atp_players import updated_players



def main_create_test(crawl_test):
    
    url = "https://www.atpworldtour.com/en/scores/current/"
    
    if not crawl_test:
        if not os.path.isfile(os.environ["DATA_PATH"] + "/test/test_{0}.csv".format(datetime.now().strftime("%Y-%m-%d"))):
            print(" new test set must be created because the one for today is missing {0}".format(datetime.now().strftime("%Y-%m-%d")))
        else:
            return pd.read_csv((os.environ["DATA_PATH"] + "/test/test_{0}.csv".format(datetime.now().strftime("%Y-%m-%d"))))
        
    ### extract new_data
    print("[0] extract new_data as test data")
    new_data = extract_futur_data(url)
   
    ### check_all_players of new_data are there 
    print("[1] update database of players base on new crawling")
    updated_players(new_data)
    
    ### clean extraction 
    print("[2] clean this information")
    clean_matches = clean(new_data, url)
    
    ### add elo
    print("[3] add elo ranking")
    latest_data= import_data()
    clean_matches_elo = add_elo(clean_matches, latest_data.loc[latest_data["target"]==1])
    
    ### add real added value stats
    print("[4] add added value stats")
    clean_matches_elo = calculate_stats(clean_matches_elo, latest_data.loc[latest_data["target"]==1])
    clean_matches_elo.to_csv(os.environ["DATA_PATH"] + "/test/test_{0}.csv".format(datetime.now().strftime("%Y-%m-%d")), index = False)


if __name__ =="__main__":
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"
    matches = main_create_test()
    