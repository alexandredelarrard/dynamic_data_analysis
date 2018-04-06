# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 16:05:58 2018

@author: JARD
"""

import numpy as np
from datetime import datetime, timedelta
import pandas as pd
#import tqdm
import time

def elo_diff(A, B):
    """
    Calculate expected score of A in a match against B

    :param A: Elo rating for player A
    :param B: Elo rating for player B
    """
    
    return 1 / (1 + 10 ** ((B - A) / 400))


def expected(sub_data, player, elo_player):
    
    real = sum((sub_data["Winner"] == player)*1)
    elo_other_player = list(sub_data.loc[sub_data["Winner"] == player, "elo2"]) + list(sub_data.loc[sub_data["Loser"] == player, "elo1"])
    expect = elo_diff(np.array([elo_player]*len(elo_other_player)), np.array(elo_other_player)).sum()
    max_date  = max(sub_data["Date"])
    
    return expect, real, max_date


def calculate_k(nbr_match):
    return 250/(nbr_match + 5)**0.4


def elo(old, exp, score, k):
    """
    Calculate the new Elo rating for a player

    :param old: The previous Elo rating
    :param exp: The expected score for this match
    :param score: The actual score for this match
    :param k: The k-factor for Elo (default: 32)
    """

    return old + k * (score - exp)


def update_end_of_season(data, year_ref, mean_elo=1000):
    """Regression towards the mean
    
    Following 538 nfl methods
    https://fivethirtyeight.com/datalab/nfl-elo-ratings-are-back/
    """
    diff_from_mean = (data.loc[data["year"] == year_ref, "elo1"] + data.loc[data["year"] == year_ref, "elo2"])/2 - mean_elo
    data.loc[data["year"] == year_ref, "elo1"] -= diff_from_mean/3
    data.loc[data["year"] == year_ref, "elo2"] -= diff_from_mean/3
    
    return data


def calculate_elo(data):
    
    data = data[["Date", "winner_id", "loser_id", "surface"]].copy()
    
    data["elo1"] = 1500
    data["elo2"] = 1500
    
    data["elo1_surface"] = 1500
    data["elo2_surface"] = 1500

    print(" Calculate elo for each player ")
    
    for i in range(len(data)):
        
        sub_data = data.iloc[i]

        elo_winner = sub_data["elo1"]
        elo_loser  = sub_data["elo2"]
        elo_winner_surface = sub_data["elo1_surface"]
        elo_loser_surface  = sub_data["elo2_surface"]
        
        index_filter = data.index <= i
        
        ##### winner elo
        player = sub_data["winner_id"]
        nbr_seen = data.loc[((data["winner_id"] ==  player) | (data["loser_id"] ==  player)) & (index_filter)]
        k_winner = calculate_k(nbr_seen.shape[0])
        new_elo = elo(elo_winner, elo_diff(elo_winner, elo_loser), 1, k=k_winner)
        
        filter_win = (data["winner_id"] == player)&(data.index > i)
        filter_lose = (data["loser_id"] == player)&(data.index > i)
        
        data.loc[filter_win, "elo1"] = new_elo
        data.loc[filter_lose, "elo2"] = new_elo
        
        #### winner elo per surface
        nbr_seen_surface = nbr_seen.loc[(nbr_seen["surface"] ==  sub_data["surface"])].shape[0]
        k_winner = calculate_k(nbr_seen_surface)
        new_elo = elo(elo_winner_surface, elo_diff(elo_winner_surface, elo_loser_surface), 1, k=k_winner)
        
        data.loc[(filter_win) & (data["surface"] ==  sub_data["surface"]), "elo1_surface"] = new_elo
        data.loc[(filter_lose) & (data["surface"] ==  sub_data["surface"]), "elo2_surface"] = new_elo
        
        ##### loser elo
        player = sub_data["loser_id"]
        nbr_seen = data.loc[((data["winner_id"] ==  player) | (data["loser_id"] ==  player))&(index_filter)]
        k_loser = calculate_k(nbr_seen.shape[0])
        new_elo = elo(elo_loser, elo_diff(elo_loser, elo_winner), 0, k=k_loser)
        
        filter_win = (data["winner_id"] == player)&(data.index > i)
        filter_lose = (data["loser_id"] == player)&(data.index > i)
        
        data.loc[filter_win, "elo1"] = new_elo
        data.loc[filter_lose, "elo2"] = new_elo
        
        #### loser elo per surface
        nbr_seen_surface = nbr_seen.loc[(nbr_seen["surface"] ==  sub_data["surface"])].shape[0]
        k_loser = calculate_k(nbr_seen_surface)
        new_elo = elo(elo_loser_surface, elo_diff(elo_loser_surface, elo_winner_surface), 0, k=k_loser)
        
        data.loc[(filter_win)&(data["surface"] ==  sub_data["surface"]), "elo1_surface"] = new_elo
        data.loc[(filter_lose)&(data["surface"] ==  sub_data["surface"]), "elo2_surface"] = new_elo
        
    return data[["Date", "winner_id", "loser_id", "elo1", "elo2", "elo1_surface", "elo2_surface"]]


def merge_data_elo(data):
    
    t0 = time.time()
    elos_extracted = calculate_elo(data)
    data["prob_elo"] = 1 / (1 + 10 ** ((elos_extracted["elo2"] - elos_extracted["elo1"]) / 400))
    data["prob_elo_surface"] = 1 / (1 + 10 ** ((elos_extracted["elo2_surface"] - elos_extracted["elo1_surface"]) / 400))
    
    data["elo1"] = elos_extracted["elo1"]
    data["elo2"] = elos_extracted["elo2"]
    data["elo1_surface"] = elos_extracted["elo1_surface"]
    data["elo2_surface"] = elos_extracted["elo2_surface"]
    
    print("[{0}s] 7) Calculate Elo ranking overall/surface ".format(time.time() - t0))
    
    return data
    
    