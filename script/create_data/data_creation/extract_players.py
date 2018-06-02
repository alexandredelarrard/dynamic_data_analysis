# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:39:18 2018

@author: User
"""

import pandas as pd
import os
from dateutil import relativedelta

def make_players():
    
    players1 = pd.read_csv(os.environ["DATA_PATH"] + "/clean_datasets/players/old/players_desc.csv")
    del players1["Unnamed: 12"]
    players1["DOB"] = pd.to_datetime(players1["DOB"], format = "%d/%m/%Y")
    
    players2 = pd.read_csv(os.environ["DATA_PATH"] + "/clean_datasets/players/old/players_desc_V2.csv")
    players2["DOB"] = pd.to_datetime(players2["DOB"], format = "%Y-%m-%d")
    players = pd.concat([players1, players2], axis=0)
    
    players["Birth place"] =  players["Birth place"].apply(lambda x : x.lstrip().rstrip())
    players["Strong_hand"] =  players["Strong_hand"].apply(lambda x : x.lstrip().rstrip())
    players["Nationality place"] =  players["Nationality"].apply(lambda x : x.lstrip().rstrip())
   
    ### fillin missing values for turned pro as birth +18yo 490/2025
    players.loc[pd.isnull(players["Turned pro"]),"Turned pro"]  =  players.loc[pd.isnull(players["Turned pro"]), "DOB"].dt.year + 18
    
    ### fillin missing weight and height as average of birth place players without missing values
    aggregate_weight = players.loc[~pd.isnull(players["Weight"]), ["Birth place", "Weight"]].groupby("Birth place").mean().astype(int)
    aggregate_height = players.loc[~pd.isnull(players["Height"]), ["Birth place", "Height"]].groupby("Birth place").mean().astype(int)
    
    players.loc[pd.isnull(players["Weight"]), "Weight"] = players.loc[pd.isnull(players["Weight"]), "Birth place"].map(aggregate_weight["Weight"])
    players.loc[pd.isnull(players["Height"]), "Height"] = players.loc[pd.isnull(players["Height"]), "Birth place"].map(aggregate_height["Height"])
    
    ### if remaining nans : replace by tennismen average
    players.loc[pd.isnull(players["Weight"]), "Weight"] = players.loc[~pd.isnull(players["Weight"]), "Weight"].mean()
    players.loc[pd.isnull(players["Height"]), "Height"] = players.loc[~pd.isnull(players["Height"]), "Height"].mean()
    
    players.loc[players["Birth place"] =="USA", "Birth place"] = "United States"
    
    players.to_csv(os.environ["DATA_PATH"] + "/clean_datasets/players/players_desc.csv", index= False)
    
    

def import_players():
    players = pd.read_csv(os.environ["DATA_PATH"] + "/clean_datasets/players/players_desc.csv")
    players = players[["Players_ID", "Player_Name", "DOB", "Turned pro", "Weight", "Height", "Nationality", "Birth place", "Strong_hand"]]
    return players.reset_index(drop=True)
    

def dates(x):
    
    diff = relativedelta.relativedelta(x[0], x[1])
    
    years = diff.years
    reste = (diff.months *30 + diff.days) / 365
    
    return years + reste


def fillin_missing_values(data):
    
    data_merged = data.copy()
    data_merged["winner_age"] = data_merged[["Date", "DOB_w"]].apply(lambda x : dates(x), axis=1)
    data_merged["loser_age"] = data_merged[["Date", "DOB_l"]].apply(lambda x : dates(x), axis=1)
    
    data_merged["winner_ht"] = data_merged["Height_w"].tolist()
    data_merged["loser_ht"] = data_merged["Height_l"].tolist()
    
    data_merged["winner_hand"] = data_merged["Strong_hand_w"].tolist()
    data_merged["loser_hand"] = data_merged["Strong_hand_l"].tolist()
    
    #### turned pro as  age
    data_merged["Turned pro_w"] = data_merged["Turned pro_w"] - data_merged["DOB_w"].dt.year
    data_merged["Turned pro_l"] = data_merged["Turned pro_l"] - data_merged["DOB_l"].dt.year
    
    data_merged = data_merged.drop(["Height_w", "Height_l", "Strong_hand_l", "Strong_hand_w",
                                    "Birth place_l", "Birth place_w", "Strong_hand_l", "Strong_hand_w", "Nationality_w", "Nationality_l"], axis=1)
    
    return data_merged


def merge_atp_players(data_merge):
    
    data = data_merge.copy()
    players = import_players()
    
    data_merged = pd.merge(data, players, left_on = "winner_id", right_on="Players_ID", how= "left")
    data_merged = pd.merge(data_merged, players, left_on = "loser_id", right_on="Players_ID",  how= "left", suffixes = ["_w","_l"])
    
    data_merged = data_merged.sort_values("ATP_ID")
    
    data_merged_missing_added = fillin_missing_values(data_merged)
    
    return data_merged_missing_added
    
