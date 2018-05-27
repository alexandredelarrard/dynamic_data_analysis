# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 16:05:58 2018

@author: JARD
"""

import time
import numpy as np
import tqdm

def elo_diff(A, B):
    """
    Calculate expected score of A in a match against B

    :param A: Elo rating for player A
    :param B: Elo rating for player B
    """
    
    return 1 / (1 + 10 ** ((B - A) / 400))


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


def calculate_elo(data):
    """
    columns : "Date", "winner_id", "loser_id", "elo1", "elo2"
    """
    
    data["elo1"] = 1500
    data["elo2"] = 1500
    data = data[["Date", "winner_id", "loser_id", "elo1", "elo2"]].copy()
    data["index"] = range(len(data))
    data = np.array(data)
    
    print(" Calculate elo for each player ")
    
    for i in tqdm.tqdm(range(len(data))):
        
        sub_data = data[i, :]

#        elo_winner_surface = sub_data["elo1_surface"]
#        elo_loser_surface  = sub_data["elo2_surface"]
        
        index_past = data[:,0]  <= sub_data[0]
        index_futur = data[:,0] > sub_data[0]
        
        ##### winner elo
        nbr_seen = data[((data[:,1] ==  sub_data[1]) | (data[:,2] ==  sub_data[1]))&(index_past)].shape[0]
        k_winner = calculate_k(nbr_seen)
        new_elo = elo(sub_data[3], elo_diff(sub_data[3], sub_data[4]), 1, k=k_winner)

        data[(data[:,1] == sub_data[1])&(index_futur), 3] = new_elo
        data[(data[:,2] == sub_data[1])&(index_futur), 4] = new_elo
        
        #### winner elo per surface
#        nbr_seen_surface = nbr_seen.loc[(nbr_seen["surface"] ==  sub_data["surface"])].shape[0]
#        k_winner = calculate_k(nbr_seen_surface)
#        new_elo = elo(elo_winner_surface, elo_diff(elo_winner_surface, elo_loser_surface), 1, k=k_winner)
#        
#        data.loc[(filter_win) & (data["surface"] ==  sub_data["surface"]), "elo1_surface"] = new_elo
#        data.loc[(filter_lose) & (data["surface"] ==  sub_data["surface"]), "elo2_surface"] = new_elo
        
        ##### loser elo
        nbr_seen = data[((data[:,1] ==  sub_data[2]) | (data[:,2] ==  sub_data[2]))&(index_past)].shape[0]
        k_loser = calculate_k(nbr_seen)
        new_elo = elo(sub_data[4], elo_diff(sub_data[4], sub_data[3]), 0, k=k_loser)
        
        data[(data[:,1] == sub_data[2])&(index_futur), 3] = new_elo
        data[(data[:,2] == sub_data[2])&(index_futur), 4] = new_elo
        
        #### loser elo per surface
#        nbr_seen_surface = nbr_seen.loc[(nbr_seen["surface"] ==  sub_data["surface"])].shape[0]
#        k_loser = calculate_k(nbr_seen_surface)
#        new_elo = elo(elo_loser_surface, elo_diff(elo_loser_surface, elo_winner_surface), 0, k=k_loser)
#        
#        data.loc[(filter_win)&(data["surface"] ==  sub_data["surface"]), "elo1_surface"] = new_elo
#        data.loc[(filter_lose)&(data["surface"] ==  sub_data["surface"]), "elo2_surface"] = new_elo
        
    return data[:,3:5]


def merge_data_elo(data):
    
    t0 = time.time()
    elos_extracted = calculate_elo(data)
    data["prob_elo"] = 1 / (1 + 10 ** ((elos_extracted[:,1] - elos_extracted[:,0]) / 400))
#    data["prob_elo_surface"] = 1 / (1 + 10 ** ((elos_extracted["elo2_surface"] - elos_extracted["elo1_surface"]) / 400))
    
    data["elo1"] = elos_extracted[:,0]
    data["elo2"] = elos_extracted[:,1]
#    data["elo1_surface"] = elos_extracted["elo1_surface"]
#    data["elo2_surface"] = elos_extracted["elo2_surface"]
    
#    data["elo_answer_surface"] = 0
#    data.loc[data["prob_elo_surface"] >=0.5, "elo_answer_surface"] = 1
    data["elo_answer"] = 0
    data.loc[data["prob_elo"] >=0.5, "elo_answer"] = 1
    
    for i in range(2014,2019):
        print("[ELO] Error {0} is {1}".format(i, 1 - sum(1 - data.loc[data["Date"].dt.year == i, "elo_answer"])/len(data.loc[data["Date"].dt.year == i, "elo_answer"])))
#        print("[ELO SURFACE] Error {0} is {1}".format(i, 1 - sum(1 - data.loc[data["Date"].dt.year == i, "elo_answer_surface"])/len(data.loc[data["Date"].dt.year == i, "elo_answer_surface"])))
            
    print("[{0}s] 7) Calculate Elo ranking overall/surface ".format(time.time() - t0))
    
    return data

