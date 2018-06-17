# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 14:26:57 2018

@author: User
"""

import os
import pandas as pd
import numpy as np
from datetime import timedelta
import glob

from create_train.data_creation.create_variables import create_basic_features
from create_train.data_creation.extract_players import import_players
from create_train.data_creation.create_statistics_historyV2 import create_stats, correlation_subset_data
from create_train.data_creation.create_elo_rankingV2 import fill_latest_elo
from create_train.utils.utils_data_prep import dates, currency_prize,homogenize_prizes,extract_rank_and_match,liste1_extract


def clean(matches, url):
    
    clean = pd.DataFrame([])
    
    list_info1 = matches["tourney"].apply(lambda x: liste1_extract(x))
    clean["draw_size"] = matches["draw"].apply(lambda x : int(x.split()[1]))
    clean["tourney_end_date"] = pd.to_datetime(list(list(zip(*list_info1))[0]), format="%Y.%m.%d") 
    clean["tourney_date"] = pd.to_datetime(list(list(zip(*list_info1))[1]), format="%Y.%m.%d")
    clean["tourney_country"] = list(list(zip(*list_info1))[2])
    clean["tourney_name"] = list(list(zip(*list_info1))[3])
    clean["tourney_city"] = list(list(zip(*list_info1))[4])
    clean["masters"] = matches["masters"]
    clean["best_of"] = np.where(clean["masters"] == "grandslam", 5, 3)
    clean["Date"] =  pd.to_datetime(matches["Date"], format= "%A, %B %d, %Y")
    clean["tourney_year"] = clean["Date"].dt.year
    clean["round"] = matches["round"].apply(lambda x: x.strip())
    clean["tourney_id_wo_year"] =  matches["url"].apply(lambda x : x.split("/")[-2])
    clean["tourney_id_wo_year"] = "_" + clean["tourney_id_wo_year"]
    clean["tourney_id"] = clean["tourney_year"].astype(str) + "-" + clean["tourney_id_wo_year"]
    clean["surface"] = matches["surface"]
    clean["prize"] = matches["prize"]
    clean["winner_name"] = matches["8"].apply(lambda x : x.lower().replace("-"," ").strip())
    clean["loser_name"] = matches["9"].apply(lambda x : x.lower().replace("-"," ").strip())
    
    # =============================================================================
    #     ### add indoor flag
    # =============================================================================
    tournament = pd.read_csv(os.environ["DATA_PATH"] + "/clean_datasets/tournament/tourney.csv", encoding = "latin1")[["tourney_id", "indoor_flag"]]
    clean = pd.merge(clean, tournament, on="tourney_id", how = "left")
    
    ### to modify
    clean["indoor_flag"] = np.where(pd.isnull(clean["indoor_flag"]), "Outdoor", clean["indoor_flag"])
    
    # =============================================================================
    #     ### take care currency 
    # =============================================================================
    count = matches["prize"].apply(lambda x : currency_prize(x))
    clean["Currency"] = list(list(zip(*count))[0])
    clean["Currency"] = np.where(clean["Currency"].isin(["euros","€"]), "euro", clean["Currency"])
    clean["Currency"] = np.where(clean["Currency"].isin(["AUS$","AU$","A$"]), "AU$", clean["Currency"])
    clean["Currency"] = np.where(clean["Currency"].isin(["A£","'Â£'","£"]), "£", clean["Currency"])
    
    # =============================================================================
    #     ### homogenize price
    # =============================================================================
    clean["prize"] = list(list(zip(*count))[1])
    currency = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/tournament/currency_evolution.csv")
    clean["prize"] = clean[["prize", "Currency", "tourney_year"]].apply(lambda x: homogenize_prizes(x, currency), axis=1)
    del clean["tourney_year"]
    
    # =============================================================================
    #     #### merge player id 
    # =============================================================================
    players = import_players()
    
    clean = pd.merge(clean, players, left_on = "winner_name", right_on = "Player_Name", how = "left")
    clean = pd.merge(clean, players, left_on = "loser_name", right_on = "Player_Name", how = "left", suffixes= ("_w","_l"))
    clean = clean.drop(["Player_Name_w", "Player_Name_l", "Birth place_l", "Birth place_w"], axis=1)
    clean["DOB_w"] = pd.to_datetime(clean["DOB_w"], format="%Y-%m-%d")
    clean["DOB_l"] = pd.to_datetime(clean["DOB_l"], format="%Y-%m-%d")
    
    clean = clean.rename(columns = {"Height_l": "loser_ht", 
                                    "Height_w": "winner_ht", 
                                    'Strong_hand_w': "winner_hand", 
                                    'Strong_hand_l': "loser_hand",
                                    'Nationality_w':"winner_ioc",
                                    'Nationality_l':"loser_ioc"})
    
    clean["winner_age"] = clean[["Date", "DOB_w"]].apply(lambda x : dates(x), axis=1)
    clean["loser_age"]  = clean[["Date", "DOB_l"]].apply(lambda x : dates(x), axis=1)
    
    # =============================================================================
    #     #### add rank data into it
    # =============================================================================
    files_rank = glob.glob(os.environ["DATA_PATH"] + "/brute_info/atp_ranking/*.csv")
    files_df = pd.DataFrame(np.transpose([files_rank, [pd.to_datetime(os.path.splitext(os.path.basename(x))[0], format = "%Y-%m-%d") for x in files_rank]]), columns = ["file", "Date"])
    files_rank = files_df.loc[files_df["Date"] >= pd.to_datetime(clean["Date"].min(), format = "%Y-%m-%d") - timedelta(days=15)]
    
    for i, f in enumerate(files_rank["file"].tolist()):
        if i ==0:
            rk_data = pd.read_csv(f)
            rk_data["Date"] = files_rank.iloc[i,1]
        else:
            new_data = pd.read_csv(f)
            new_data["Date"] = files_rank.iloc[i,1]
            rk_data = pd.concat([rk_data, new_data],axis =0)
            
    rk_data= rk_data.reset_index(drop=True)
    rk_data["player_rank"] = rk_data["player_rank"].str.replace("T","").astype(int)
    rk_data["Player_name"] = rk_data["Player_name"].apply(lambda x : x.replace("-"," ").lower())
    
    count = clean[["Date", "winner_name", "loser_name"]].apply(lambda x: extract_rank_and_match(x, rk_data), axis =1)["Date"]
    
    for i, col in enumerate(["winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points"]):
        clean[col]  = list(list(zip(*count))[i])
        
    clean = clean.rename(columns = {"Players_ID_w":"winner_id", "Players_ID_l":"loser_id"})
    clean = clean.sort_values(["Date", "tourney_id"]).reset_index(drop=True)
    
    # =============================================================================
    #     ### additionnal features to create
    # =============================================================================
    clean = create_basic_features(clean)
    
    return clean


def add_elo(clean_matches, latest_data):
    
    matches_elo, dico_players_nbr = fill_latest_elo(latest_data, clean_matches) 
    return matches_elo

    
def calculate_stats(clean_matches_elo, latest_data):

    # =============================================================================
    #     ### calculate the statistics on it
    # =============================================================================
    cols_for_stat_calc, correlation_surface, correlation_time = correlation_subset_data(clean_matches_elo, redo = False)

    liste_dataframe = [latest_data[cols_for_stat_calc], correlation_surface, correlation_time]
    clean_matches_elo_stats = create_stats(clean_matches_elo.reset_index(drop=True), liste_dataframe)
        
    return clean_matches_elo_stats