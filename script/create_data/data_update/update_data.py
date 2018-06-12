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

from create_data.data_update.clean_updated_data import clean_extract
from create_data.data_creation.create_statistics_historyV2 import get_correlations, create_stats
from create_data.data_creation.create_elo_rankingV2 import fill_latest_elo, calculate_elo_over_the_road

sys.path.append(r"C:\Users\User\Documents\tennis")
from crawling.crawling_additionnal_data import extract_additionnal_data
from crawling.crawling_atp_ranking import atp_crawl

def import_data():
    
    try:
        path = os.environ["DATA_PATH"] + "/clean_datasets/overall/updated/latest/total_dataset_modelling.csv"
        latest_data = pd.read_csv(path)
        print("from updated/latest")
    except Exception:
        print("from stable/historical")
        path = os.environ["DATA_PATH"] + "/clean_datasets/overall/stable/hictorical_origin/total_dataset_modelling.csv"
        latest_data = pd.read_csv(path)
        pass
    
    return latest_data


def update_stable():
    
    latest_data = import_data()
        
    latest_data = latest_data.loc[latest_data["target"] == 1]
    latest_data = latest_data.sort_values(["tourney_date", "tourney_name"])
    
    latest = {"Date": latest_data["tourney_date"].max()}
    
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
    extra = clean_extract(latest)
    print("time for cleaning the crawled data {0}".format(time.time() - t0))
    
    ### calculate elo
    latest_data["Date"] = pd.to_datetime(latest_data["Date"], format = "%Y-%m-%d")
    latest_data["tourney_date"] = pd.to_datetime(latest_data["tourney_date"], format = "%Y-%m-%d")
    latest_data["DOB_w"] = pd.to_datetime(latest_data["DOB_w"], format = "%Y-%m-%d")
    latest_data["DOB_l"] = pd.to_datetime(latest_data["DOB_l"], format = "%Y-%m-%d")
    
    t0 = time.time()
    additionnal_data, dico_players_nbr = fill_latest_elo(latest_data, extra)
    extra = calculate_elo_over_the_road(additionnal_data, dico_players_nbr)
    print("Calculate elo for new match {0}".format(time.time() - t0))
    
    # =============================================================================
    #     ### calculate the statistics on it
    # =============================================================================
    correlation_surface, correlation_time = get_correlations(extra, redo = False)
    calculate_stats = ['Date', 'winner_id', 'loser_id', "surface", 'minutes', 'best_of', "winner_rank", 'loser_rank', 
                       'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
                       'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced',
                       'w_1st_srv_ret_won','w_2nd_srv_ret_won', 'w_bp_converted', 'w_total_srv_won', 'w_total_ret_won', 
                       'l_1st_srv_ret_won', 'l_2nd_srv_ret_won', 'l_bp_converted', 'l_total_srv_won', 'l_total_ret_won',
                       'w_tie-breaks_won', 'l_tie-breaks_won', 'Nbr_tie-breaks', "N_set", 'l_total_pts_won', 'w_total_pts_won',
                       'tourney_id_wo_year', "id_round"]

    for i, tourney in enumerate(extra["tourney_id"].unique()):
        print(" additionnal stats on {0} start date {1} end date {2}".format(tourney, extra.loc[extra["tourney_id"] == tourney, "tourney_date"].unique()[0], extra.loc[extra["tourney_id"] == tourney, "tourney_end_date"].unique()[0]))
        liste_dataframe = [np.array(latest_data.loc[calculate_stats]), correlation_surface, correlation_time]
        addi = create_stats(extra.loc[extra["tourney_id"] == tourney].reset_index(drop=True), liste_dataframe)
        
        if i ==0:
            extraction_total = addi
        else:
            extraction_total = pd.concat([extraction_total, addi], axis =0)
            
    extraction_total = extraction_total.reset_index(drop=False)
                   
    extraction_total.to_csv(os.environ["DATA_PATH"] + "/clean_datasets/overall/updated/extracted/extraction_clean.csv", index = False)
    extraction_total.to_csv(os.environ["DATA_PATH"] + "/clean_datasets/overall/stable/all_extractions/extraction_clean_%s.csv"%str(max_date), index = False)
    
    # =============================================================================
    #     ##### merge with new updated data and new update, move the previous most updated data to old folder
    # =============================================================================
    latest_data = import_data()
    new_data_modelling = pd.concat([latest_data, extraction_total], axis=0)
    files_already_there = glob.glob(os.environ["DATA_PATH"] + "/clean_datasets/overall/updated/latest/*.csv")
    for f in files_already_there: 
        os.rename(f, os.environ["DATA_PATH"] + "/clean_datasets/overall/updated/latest/old/{0}_total_dataset_modelling.csv".format(latest["Date"]))
       
    new_data_modelling.to_csv(os.environ["DATA_PATH"] + "/clean_datasets/overall/updated/latest/total_dataset_modelling.csv", index = False)
    
    return new_data_modelling

if __name__ == "__main__":
    
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"
    extra = update_stable()
    