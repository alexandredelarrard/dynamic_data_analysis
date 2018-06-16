# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 11:09:55 2018

@author: User
"""

import pandas as pd
import os

from plot_lib import var_vs_target
   

def hist_cols(data):    
    for col in ['tourney_name', 'surface', 'draw_size', 'tourney_level',
           'match_num', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age', 'winner_rank',
           'winner_rank_points', 'loser_hand',
           'loser_ht', 'loser_age', 'loser_rank', 'loser_rank_points',
           'best_of', 'round', 'minutes', 'w_ace', 'w_df', 'w_svpt',
           'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
           'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms',
           'l_bpSaved', 'l_bpFaced', 'tourney_end_date', 'prize', 'masters',
           'Currency', 'indoor_flag', 'tourney_country', 'tourney_city', 'Date',
           'diff_days_since_stop', 'missing_stats', 'DOB_w', 'Turned pro_w', 'Weight_w', 'DOB_l', 'Turned pro_l', 'Weight_l', 'elo1',
           'elo2']:
    
        var_vs_target(data, 'prob_elo', col, bins=30, normalize = False)
        
# =============================================================================
# ### check tourney dataset        
# =============================================================================
def check_tourney():
    tourney = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/tournament/tourney.csv")
    tourney["tourney_date"] = pd.to_datetime(tourney["tourney_date"], format="%Y-%m-%d")
    tourney["tourney_end_date"] = pd.to_datetime(tourney["tourney_end_date"], format="%Y-%m-%d")
    
    tourney["length"]  = (tourney["tourney_end_date"] - tourney["tourney_date"]).dt.days
    
    #### worl team cup takes 35 days
    tourney["length"].hist(bins = 30, range = (-1,tourney["length"].max()))
    
# =============================================================================
# ### players       
# =============================================================================
def check_players():
    players = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/players/players_desc.csv")
    players["DOB"] = pd.to_datetime(players["DOB"], format="%Y-%m-%d")
    players["turned_pro_age"]  = players["Turned pro"] - players["DOB"].dt.year
    
    #### worl team cup takes 35 days
    players["turned_pro_age"].hist(bins = 30, range = (0,players["turned_pro_age"].max()))
    
    
# =============================================================================
# ### match stats     
# =============================================================================
def check_stats(data):
    
    #### no service game
    a = data.loc[data["l_SvGms"]<3][["tourney_name","Date","w_SvGms","l_svpt","w_svpt","score"]]
    b=  data.loc[data["w_SvGms"]<3][["tourney_name","Date","l_SvGms", "l_svpt","w_svpt","score"]]
    
if __name__ == "__main__":
    
        
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"
    data=    pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/matches_elo_V1.csv")
    hist_cols(data)     