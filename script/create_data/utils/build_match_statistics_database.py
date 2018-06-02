# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 20:29:59 2018

@author: User
"""

import pandas as pd
import numpy as np

import sys 

sys.path.append(r"C:\Users\User\Documents\tennis\crawling")
from crawling_additionnal_data import extract_additionnal_data


def clean_stat(new_data_orig, dd):
    
    missing_stats = new_data_orig.copy()
    
    missing_stats = pd.merge(new_data_orig, dd, on ="ID", how= "left")
    
    ### take care of time variable to have match length in minutes
    missing_stats["Time"] = missing_stats["time"].fillna("0:0:0").apply(lambda x : pd.to_datetime(str(x).replace("Time: ",""), format = "%H:%M:%S").hour * 60 + pd.to_datetime(str(x).replace("Time: ",""), format = "%H:%M:%S").minute)
    missing_stats["Time"] = missing_stats["Time"].replace(0, np.nan)
    
    #### keep matched values
    missing_stats = missing_stats.loc[~pd.isnull(missing_stats["player1"])]
    
    missing_stats["w_ace"] = missing_stats["aces_player1"].astype(int)
    missing_stats["l_ace"] = missing_stats["aces_player2"].astype(int)
    missing_stats["w_df"] = missing_stats["df_player1"].astype(int)
    missing_stats["l_df"] = missing_stats["df_player2"].astype(int)
    
    missing_stats["w_svpt"] = missing_stats["1serv_player1"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[1])
    missing_stats["l_svpt"] = missing_stats["1serv_player2"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[1])
    missing_stats["w_1stIn"] = missing_stats["1serv_player1"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    missing_stats["l_1stIn"] = missing_stats["1serv_player2"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    missing_stats["w_1stWon"] = missing_stats["1serv_won_player1"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    missing_stats["l_1stWon"] = missing_stats["1serv_won_player2"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    missing_stats["w_2ndWon"] = missing_stats["2serv_won_player1"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    missing_stats["l_2ndWon"] = missing_stats["2serv_won_player2"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    missing_stats["w_SvGms"] = missing_stats["serv_games_player1"].astype(int)
    missing_stats["l_SvGms"] = missing_stats["serv_games_player2"].astype(int)
    
    missing_stats["w_bpSaved"] = missing_stats["bp_saved_player1"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    missing_stats["l_bpSaved"] = missing_stats["bp_saved_player2"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    missing_stats["w_bpFaced"] = missing_stats["bp_saved_player1"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[1])
    missing_stats["l_bpFaced"] = missing_stats["bp_saved_player2"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[1])
    missing_stats["minutes"] = missing_stats["Time"].astype(int)
    
    return missing_stats[['w_ace', 'w_df', 'w_svpt', 'w_1stIn',
                           'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace',
                           'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms',
                           'l_bpSaved', 'l_bpFaced', 'minutes']]
    
    
def match_stats_main(data_atp, redo = False):
    
    if redo:
        sub_data = data_atp.loc[(pd.isnull(data_atp["w_ace"]))|(data_atp["w_SvGms"] == 0)|(data_atp["l_SvGms"] == 0)]
        print("[Crawl wrong/missing stats] data wrong has shape {0}".format(sub_data.shape))
        
        liste = sub_data["tourney_id"].apply(lambda x : x.split("-")[1] + "/" + x.split("-")[0])
        latest = {"Date":sub_data["Date"].min(), "liste_tourney" :  list(set(liste))}
        new_data = extract_additionnal_data(latest)
        
        ### Create stats needed for total data 
        missing_stats = clean_stat(new_data)
        missing_stats.to_csv(r"C:\Users\User\Documents\tennis\data\brute_info\historical\correct_missing_values\missing_match_stats.csv", index= False)
        
    else:
        missing_stats = pd.read_csv(r"C:\Users\User\Documents\tennis\data\brute_info\historical\correct_missing_values\missing_match_stats.csv")
    
    return missing_stats
    
if __name__ == "__main__":
    missing_stats = match_stats_main(data_atp, redo = True)
    
    