# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:38:37 2018

@author: User
"""

import pandas as pd
import os
import sys
import time
from datetime import datetime, timedelta
import tqdm

sys.path.append(r"C:\Users\User\Documents\tennis\dynamic_data_analysis\script")
from create_train.data_update.clean_updated_data import clean_extract
from create_train.data_creation.create_statistics_historyV2 import create_stats, correlation_subset_data
from create_train.data_creation.create_elo_rankingV2 import fill_latest_elo, calculate_elo_over_the_road

sys.path.append(r"C:\Users\User\Documents\tennis")
from crawling.crawling_additionnal_data import extract_additionnal_data


def update_stable(historical_data):
   
    latest_data = historical_data.loc[historical_data["target"] == 1].copy()
    latest_data = latest_data.sort_values(["tourney_date", "tourney_name"])
    
    #### only crawl tourney id we don't have or we have without reaching final
    already_tourney_id = latest_data.loc[(pd.to_datetime(latest_data["tourney_end_date"]) + timedelta(days = 2) < (pd.to_datetime(latest_data["tourney_date"].max()) ))&
                                  (pd.to_datetime(latest_data["tourney_end_date"]).dt.year == pd.to_datetime(latest_data["tourney_date"].max()).year), "tourney_id"].unique()
    
    latest = {"Date": latest_data["tourney_date"].max(),
              "already_tourney_id" : already_tourney_id}
    
    ### crawl data   
    t0 = time.time()
    extract_additionnal_data(latest)
    print("time for atp latest games crawling {0}".format(time.time() - t0))
    

def clean_new_matches(historical_data):
    
    latest_data = historical_data.loc[historical_data["target"] == 1].copy()
    latest_data = latest_data.sort_values(["tourney_date", "tourney_name"])
    latest = {"Date": latest_data["tourney_date"].max()}
    
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
    #     ### calculate the statistics on it / suppress before all observation on same crawled tourney id
    # =============================================================================
    cols_stat, correlation_surface, correlation_time = correlation_subset_data(extra, redo = False)
    latest_data = latest_data.loc[~latest_data["tourney_id"].isin(list(extra["tourney_id"].unique()))]
    extra = extra.sort_values(["Date", "tourney_id"])
    i =0
    
    for tourney in tqdm.tqdm(extra["tourney_id"].unique()):
        new_to_stats = extra.loc[extra["tourney_id"] == tourney].reset_index(drop=True)
        print(" additionnal stats on {0} shape {1}".format(tourney, new_to_stats.shape[0]))
        for date in new_to_stats["Date"].unique():
            new_to_stats_date = new_to_stats.loc[new_to_stats["Date"] == date].reset_index(drop=True)
            liste_dataframe = [latest_data[cols_stat], correlation_surface, correlation_time]
            addi = create_stats(new_to_stats_date, liste_dataframe, verbose = 0)
            
            if i ==0:
                extraction_total = addi
                i =1
            else:
                extraction_total = pd.concat([extraction_total, addi], axis =0)
                
    extraction_total = extraction_total.reset_index(drop=True)
    
    if os.path.isfile(os.environ["DATA_PATH"] + "/clean_datasets/overall/stable/all_extractions/extraction_clean_{0}.csv".format(datetime.now().strftime("%Y-%m-%d"))):
        current_extra_day = pd.read_csv(os.environ["DATA_PATH"] + "/clean_datasets/overall/stable/all_extractions/extraction_clean_{0}.csv".format(datetime.now().strftime("%Y-%m-%d")))
        current_extra_day = current_extra_day.loc[~current_extra_day["tourney_id"].isin(list(extraction_total["tourney_id"].unique()))]
        extraction_total = pd.concat([current_extra_day, extraction_total],axis=0).reset_index(drop=True).drop_duplicates()
        
    extraction_total.to_csv(os.environ["DATA_PATH"] + "/clean_datasets/overall/stable/all_extractions/extraction_clean_{0}.csv".format(datetime.now().strftime("%Y-%m-%d")), index = False)
    os.remove(os.environ["DATA_PATH"] + "/clean_datasets/overall/updated/extracted/extraction_brute.csv")
    
    return extraction_total


if __name__ == "__main__":
    
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"
    extra = update_stable()
    