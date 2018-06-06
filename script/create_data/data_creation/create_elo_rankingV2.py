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


def fill_latest_elo(data, additionnal_data):
    
    data = np.array(data.loc[data["target"]==1][["Date","Players_ID_w", "Players_ID_l", "elo1", "elo2"]].copy())
    dico_players_elo = {}
    dico_players_nbr = {}
    
    players_ids = list(set(additionnal_data["Players_ID_w"].tolist() + additionnal_data["Players_ID_l"].tolist()))
    for pl in players_ids:
        sub_data = data[(data[:,1] == pl) |(data[:,2] == pl),:]
        nbr = sub_data.shape[0]
        if nbr ==0:
            print("new player {0}".format(pl))
            dico_players_elo[pl] = 1500
            dico_players_nbr[pl] = 0
        else:    
            index = np.where(sub_data[:,0]==np.max(sub_data[:,0]))
            if pl == sub_data[index,1]:
                elo = sub_data[index,3]
            else:
                elo = sub_data[index,4]
                
            dico_players_elo[pl] = elo[0][0]
            dico_players_nbr[pl] = nbr
        
    additionnal_data["elo1"] = additionnal_data["Players_ID_w"].map(dico_players_elo)
    additionnal_data["elo2"] = additionnal_data["Players_ID_l"].map(dico_players_elo)
    
    return additionnal_data, dico_players_nbr


def update_elo(data, i, dico_nbr_seen = {}):
    
    sub_data = data[i, :]

    index_past = data[:,0]  <= sub_data[0]
    index_futur = data[:,0] > sub_data[0]
    
    ##### winner elo
    if not dico_nbr_seen:
        nbr_seen = data[((data[:,1] ==  sub_data[1]) | (data[:,2] ==  sub_data[1]))&(index_past)].shape[0]
    else:
        nbr_seen = dico_nbr_seen[sub_data[1]] + 1
        
    k_winner = calculate_k(nbr_seen)
    new_elo = elo(sub_data[3], elo_diff(sub_data[3], sub_data[4]), 1, k=k_winner)

    data[(data[:,1] == sub_data[1])&(index_futur), 3] = new_elo
    data[(data[:,2] == sub_data[1])&(index_futur), 4] = new_elo
    
    ##### loser elo
    if not dico_nbr_seen:
        nbr_seen = data[((data[:,1] ==  sub_data[2]) | (data[:,2] ==  sub_data[2]))&(index_past)].shape[0]
    else:
        nbr_seen = dico_nbr_seen[sub_data[1]] + 1
        
    k_loser = calculate_k(nbr_seen)
    new_elo = elo(sub_data[4], elo_diff(sub_data[4], sub_data[3]), 0, k=k_loser)
    
    data[(data[:,1] == sub_data[2])&(index_futur), 3] = new_elo
    data[(data[:,2] == sub_data[2])&(index_futur), 4] = new_elo
    
    return data


def calculate_elo_over_the_road(data, nbr_dico):
    """
    it is used when new data crawled have to get an elo update
    -  data = crawled data
    -  nbr dico = dictionnary for all players that need an update, information is the number of time player has been seen
    """
    
    data2 = data.copy()
    data_cols = np.array(data2[["Date", "Players_ID_w", "Players_ID_l", "elo1", "elo2"]])

    print(" Calculate elo for each player ")
    for i in tqdm.tqdm(range(len(data_cols))):
        data_cols = update_elo(data_cols, i, nbr_dico)
        
    data2["elo1"] = data_cols[:,3]
    data2["elo2"] = data_cols[:,4]
        
    return data2


def calculate_elo(data):
    """
    columns : "Date", "winner_id", "loser_id", "elo1", "elo2"
    """
    
    data = data.sort_values(["tourney_date", "Date"])
    data = np.array(data[["Date", "winner_id", "loser_id", "elo1", "elo2"]].copy())

    print(" Calculate elo for each player ")
    for i in tqdm.tqdm(range(len(data))):
        data = update_elo(data, i)
        
    return data[:,3:5]


def merge_data_elo(data):
    
    t0 = time.time()
    data["elo1"] = 1500
    data["elo2"] = 1500
    
    #### calculate elo
    elos_extracted = calculate_elo(data)
    data["prob_elo"] = 1 / (1 + 10 ** ((elos_extracted[:,1] - elos_extracted[:,0]) / 400))
    
    data["elo1"] = elos_extracted[:,0]
    data["elo2"] = elos_extracted[:,1]
    
    data["elo_answer"] = 0
    data.loc[data["prob_elo"] >=0.5, "elo_answer"] = 1
    
    for i in range(2014,2019):
        print("[ELO] Error {0} is {1}".format(i, 1 - sum(1 - data.loc[data["Date"].dt.year == i, "elo_answer"])/len(data.loc[data["Date"].dt.year == i, "elo_answer"])))
            
    del data["elo_answer"]
    print("[{0}s] 7) Calculate Elo ranking overall/surface ".format(time.time() - t0))
    
    return data

