# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 14:20:34 2018

@author: User
"""

import os
import sys

os.chdir(r"C:\Users\User\Documents\tennis\dynamic_data_analysis\script")
sys.path.append(r"C:\Users\User\Documents\tennis\dynamic_data_analysis\script")
from create_test.data_creation.clean_test_data import clean, add_elo, calculate_stats
from create_train.data_update.main_create_update import create_update

sys.path.append(r"C:\Users\User\Documents\tennis\crawling")
from crawling_futur_match import extract_futur_data

def main_create_test():
    
    url = "http://www.atpworldtour.com/en/scores/current/"
    
    #### check train is up to date
    latest_data = create_update(boolean_update = True)
       
    ### extract new_data
    new_data = extract_futur_data(url)
    
    ### clean extraction 
    print("[3] clean this information")
    clean_matches = clean(new_data, url)
    
    ### add elo
    print("[4] add elo ranking")
    clean_matches_elo, latest_data = add_elo(clean_matches, latest_data)
    
    ### add real added value stats
    print("[5] add added value stats")
    clean_matches_elo = calculate_stats(clean_matches_elo, latest_data)
    
    return new_data


if __name__ =="__main__":
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"
    matches = main_create_test()
    