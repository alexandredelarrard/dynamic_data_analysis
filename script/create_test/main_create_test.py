# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 14:20:34 2018

@author: User
"""

import os
import sys
import datetime

#os.chdir(r"C:\Users\User\Documents\tennis\dynamic_data_analysis\script")
sys.path.append(r"C:\Users\User\Documents\tennis\dynamic_data_analysis\script")
from create_test.data_creation.clean_test_data import clean, add_elo, calculate_stats
from create_train.data_update.main_create_update import create_update

sys.path.append(r"C:\Users\User\Documents\tennis\crawling")
from crawling_futur_match import extract_futur_data
from crawling_atp_players import updated_players


def main_create_test(boolean_update):
    
    url = "http://www.atpworldtour.com/en/scores/current/"
    
    ### extract new_data
    print("[0] extract new_data as test data")
    new_data = extract_futur_data(url)
    
    ### check_all_players of new_data are there 
    print("[1] update database of players base on new crawling")
    updated_players(new_data)
    
    #### check train is up to date
    print("[2] check all dataset are up to date")
    latest_data = create_update(boolean_update = boolean_update)
    
    ### clean extraction 
    print("[3] clean this information")
    clean_matches = clean(new_data, url)
    
    ### add elo
    print("[4] add elo ranking")
    clean_matches_elo = add_elo(clean_matches, latest_data.loc[latest_data["target"]==1])
    
    ### add real added value stats
    print("[5] add added value stats")
    clean_matches_elo = calculate_stats(clean_matches_elo, latest_data.loc[latest_data["target"]==1])
    clean_matches_elo.to_csv(os.environ["DATA_PATH"] + "/test/test_{0}.csv".format(datetime.datetime.now().strftime("%Y-%m-%d")))


if __name__ =="__main__":
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"
    matches = main_create_test()
    