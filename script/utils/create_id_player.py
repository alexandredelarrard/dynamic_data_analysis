# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 15:55:02 2018

@author: JARD
"""

import pandas as pd
import numpy as np

def update_elo(winner_elo, loser_elo, k_factor=32):

    expected_win = expected(winner_elo, loser_elo)
    change_in_elo = k_factor * (1-expected_win)
    winner_elo += change_in_elo
    loser_elo -= change_in_elo
    
    return winner_elo, loser_elo


def update_end_of_season(elos):
    """Regression towards the mean
    
    Following 538 nfl methods
    https://fivethirtyeight.com/datalab/nfl-elo-ratings-are-back/
    """
    mean_elo = 1500
    diff_from_mean = elos - mean_elo
    elos -= diff_from_mean/3
    return elos


def expected(A, B):
    """
    Calculate expected score of A in a match against B

    :param A: Elo rating for player A
    :param B: Elo rating for player B
    """
    return 1 / (1 + 10 ** ((B - A) / 400))

def clean_merge(merg):
    
    merg["Winner"] = merg["Winner"].apply(lambda x : x.lstrip().rstrip())
    merg["Loser"] = merg["Loser"].apply(lambda x : x.lstrip().rstrip())
    
    #### get players name 
    players = set(merg["Winner"].tolist() + merg["Loser"].tolist())
    
    Players_encoding = pd.DataFrame(list(players), columns = ["player_name"])
    Players_encoding = Players_encoding.sort_values("player_name").reset_index(drop = True)
    Players_encoding["player_id"] = Players_encoding.index
    Players_encoding.index= Players_encoding["player_name"]
    del Players_encoding["player_name"]
    
    ### map encoding
    merg["Winner"] = merg["Winner"].map(Players_encoding["player_id"])
    merg["Loser"] = merg["Loser"].map(Players_encoding["player_id"])
    
    for col in ["W1", "W2", "W3", "W4", "W5"]:
        merg[col] = merg[col].replace("", "0")
        merg[col.replace("W", "L")] = merg[col.replace("W", "L")].replace(", ", "0")
    
    ## number of set won
    merg["set_won"] = merg.fillna("0").apply(lambda x : sum([(float(x["W%i"%i]) > float(x["L%i"%i]))*1 for i in range(1,6)]), axis = 1)
    
    
    ## number of set lost
    
    ### number of jeu won 
    
    ### number jeu lost
    
    

    return merg

if __name__ == "__main__":
    merg = pd.read_csv(r"C:\Users\JARD\Documents\Projects\tennis\merged.csv")
    merg = clean_merge(merg)
    
    
    
    