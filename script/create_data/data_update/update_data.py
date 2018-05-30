# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:38:37 2018

@author: User
"""

import pandas as pd
import os
import sys
import time
import numpy as np
import glob

from data_update.clean_updated_data import clean_extract
from create_data.data_creation.create_statistics_historyV2 import get_correlations, create_stats
from create_data.data_creation.create_elo_rankingV2 import update_elo

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
    max_date = atp_crawl(latest)
    print("time for atp rank crawling {0}".format(time.time() - t0))

    ### crawl data   
    t0 = time.time()
    extract_additionnal_data(latest)
    print("time for atp latest games crawling {0}".format(time.time() - t0))
    
    ### clean the crawled data
    t0 = time.time()
    extra = clean_extract()
    print("time for cleaning the crawled data {0}".format(time.time() - t0))
    
    ### calculate elo
    data = pd.read_csv(os.environ["DATA_PATH"] + "/clean_datasets/overall/stable/total_dataset_modelling.csv")
    t0 = time.time()
    extra = update_elo(extra)
    print("Calculate elo for new match {0}".format(time.time() - t0))
    
    ### calculate the statistics on it
    correlation_surface, correlation_time = get_correlations(extra, redo = False)
    calculate_stats = ['Date', 'winner_id', 'loser_id', "surface", 'minutes', 'missing_stats', "winner_rank", 'loser_rank', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
                     'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced','w_1st_srv_ret_won',
                     'w_2nd_srv_ret_won', 'w_bp_converted', 'w_total_srv_won', 'w_total_ret_won', 'l_1st_srv_ret_won', 'l_2nd_srv_ret_won', 'l_bp_converted',
                     'l_total_srv_won', 'l_total_ret_won', 'w_tie-breaks_won', 'l_tie-breaks_won', 'Nbr_tie-breaks', "N_set", 'l_total_pts_won', 'w_total_pts_won', "match_num"]

    liste_params = [np.array(data[calculate_stats]), correlation_surface, correlation_time]
    total_data = create_stats(data, liste_params)
    
    files_already_there = glob.glob(os.environ["DATA_PATH"] + "/clean_datasets/overall/updated/*.csv")
    for f in files_already_there: 
        os.rename(f, os.environ["DATA_PATH"] + "/clean_datasets/overall/updated/old/%s"%os.path.basename(f))
        
    total_data.to_csv(os.environ["DATA_PATH"] + "/clean_datasets/overall/updated/extraction_clean_%s.csv"%str(max_date))
    
    ### update total_data for modelling
    files_already_there = glob.glob(os.environ["DATA_PATH"] + "/clean_datasets/overall/stable/*.csv")
    for f in files_already_there: 
        os.rename(f, os.environ["DATA_PATH"] + "/clean_datasets/overall/stable/old/{0}_{1}".format(latest["Date"], os.path.basename(f)))
       
    new_data_modelling = pd.concat([data, total_data], axis=1)
    new_data_modelling.to_csv(os.environ["DATA_PATH"] + "/clean_datasets/overall/stable/total_dataset_modelling.csv")
    
    return new_data_modelling


if __name__ == "__main__":
    extra = update_stable()
    