# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 20:29:59 2018

@author: User
"""

import pandas as pd
import numpy as np
import os
import sys 

sys.path.append(r"C:\Users\User\Documents\tennis\crawling")
from crawling_additionnal_data import extract_additionnal_data


def liste1_extract(x):
    
    liste= eval(x)
    list2= liste[0].split()
    draw_size = liste[1].split()[1]
    end_date  = list2[-1]
    start_date = list2[-3]
    country = list2[-4]
    city = liste[0].split(",")[0].split(" ")[-1]
    name = " ".join(liste[0].split(",")[0].split(" ")[:-1])
    master = "" if liste[-1] == "atpwt" else liste[-1]
    
    return (draw_size, end_date, start_date, country, city, name, master)


def calculate_time(x):
    x = x.replace(".00","")
    try:
         x = pd.to_datetime(str(x).replace("Time: ",""), format = "%H:%M:%S")
         time = x.hour * 60 + x.minute
         
    except Exception:
       return np.nan
    
    return time


def clean_stat(extract):
    """
    clean the new data extracted from atp website
    """
    
    dico_round = {"Round of 32": "R32", "Round of 16": "R16", "Round of 64": "R64", "Round of 128":"R128", "Finals": "F", "Semi-Finals":"SF", "Quarter-Finals":"QF", "Round Robin":"RR", "Round of 96":"R96"}
   
    ex = extract.copy()
    ex.columns = [str(x) for x in ex.columns]
    clean = pd.DataFrame([])
    
    # =============================================================================
    #     ### extraction of clean info
    # =============================================================================
    list_info1 = ex["7"].apply(lambda x : liste1_extract(x))
    clean["winner_name"] = ex["8"].apply(lambda x : x.lower().replace("-"," ").lstrip().rstrip())
    clean["loser_name"] = ex["9"].apply(lambda x : x.lower().replace("-"," ").lstrip().rstrip())
    clean["tourney_date"] = pd.to_datetime(list(list(zip(*list_info1))[2]), format="%Y.%m.%d")
    clean["tourney_id"] = clean["tourney_date"].dt.year.astype(str) + "-" + ex["6"].astype(str) 
    clean["round"] = ex["64"].str.replace(" H2H", "")
    clean["round"] = clean["round"].map(dico_round)
    clean["tourney_name"] = list(list(zip(*list_info1))[5])
    
    # =============================================================================
    #     #### take care of stats
    # =============================================================================
    ex_stats = ex.iloc[:,12:]
    ex_stats.columns= [str(x) for x in range(ex_stats.shape[1])]
    ex_stats = ex_stats.rename(columns = {"0":"url", "2": "sr_player1", "4": "sr_player2", 
                        "5": "aces_player1", "7" :"aces_player2", "8":"df_player1", "10":"df_player2", 
                        "11": "1serv_player1", "13": "1serv_player2", "14": "1serv_won_player1", "16": "1serv_won_player2",
                        "17": "2serv_won_player1", "19": "2serv_won_player2", "20" : "bp_saved_player1", "22" : "bp_saved_player2",
                        "23": "serv_games_player1", "25": "serv_games_player2", "27" : "rr_player1", "29" : "rr_player2",
                        "30": "1serv_return_won_player1", "32": "1serv_return_wplayerson_player2", "33": "2serv_return_won_player1", "35": "2serv_return_won_player2",
                        "36": "bp_converted_player1", "38": "bp_converted_player2", "39": "return_games_player1", "41": "return_games_player2",
                        "53": "player1", "54" : "score_player1", "55":"player2", "56": "score_player2", "57": "time"})
    
    clean["minutes"] = ex_stats["time"].fillna("0:0:0").apply(lambda x :calculate_time(x))
    clean["minutes"] = clean["minutes"].replace(0, np.nan).astype(int)
    
    clean["w_ace"] = ex_stats["aces_player1"].astype(int)
    clean["l_ace"] = ex_stats["aces_player2"].astype(int)
    clean["w_df"] = ex_stats["df_player1"].astype(int)
    clean["l_df"] = ex_stats["df_player2"].astype(int)
    
    clean["w_svpt"] = ex_stats["1serv_player1"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[1])
    clean["l_svpt"] = ex_stats["1serv_player2"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[1])
    clean["w_1stIn"] = ex_stats["1serv_player1"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    clean["l_1stIn"] = ex_stats["1serv_player2"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    clean["w_1stWon"] = ex_stats["1serv_won_player1"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    clean["l_1stWon"] = ex_stats["1serv_won_player2"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    clean["w_2ndWon"] = ex_stats["2serv_won_player1"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    clean["l_2ndWon"] = ex_stats["2serv_won_player2"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    clean["w_SvGms"] = ex_stats["serv_games_player1"].astype(int)
    clean["l_SvGms"] = ex_stats["serv_games_player2"].astype(int)
    
    clean["w_bpSaved"] = ex_stats["bp_saved_player1"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    clean["l_bpSaved"] = ex_stats["bp_saved_player2"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    clean["w_bpFaced"] = ex_stats["bp_saved_player1"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[1])
    clean["l_bpFaced"] = ex_stats["bp_saved_player2"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[1])

    return clean
    
    
def match_stats_main(data_atp, redo = False):
    
    # =============================================================================
    #     ##### crawl missing stats and add them to missing ones
    # =============================================================================
    sub_data = data_atp.loc[((pd.isnull(data_atp["w_ace"]))|(data_atp["w_svpt"] == 0)|(data_atp["l_svpt"] == 0)
                                            | (data_atp["l_1stIn"] == 0)|(data_atp["w_1stIn"] == 0))&(data_atp["Date"].dt.year >=2000)].copy()
    
    if redo:
        print("[Crawl wrong/missing stats] data wrong has shape {0}".format(sub_data.shape))
        
        liste = sub_data["tourney_id"].apply(lambda x : x.split("-")[1] + "/" + x.split("-")[0])
        latest = {"Date":sub_data["Date"].min(), "liste_tourney" :  list(set(liste))}
        extract_additionnal_data(latest)
        
        new_data = pd.read_csv(os.environ["DATA_PATH"] + "/brute_info/historical/correct_missing_values/missing_match_stats.csv")
        
        ### Create stats needed for total data 
        missing_stats = clean_stat(new_data)
        missing_stats.to_csv(os.environ["DATA_PATH"] + "/brute_info/historical/correct_missing_values/missing_match_stats.csv", index= False)
    else:
        missing_stats = pd.read_csv(os.environ["DATA_PATh"] + "/brute_info/historical/correct_missing_values/missing_match_stats.csv")
        missing_stats["winner_name"] = missing_stats["winner_name"].apply(lambda x : x.lower().replace("-"," "))
        missing_stats["loser_name"] = missing_stats["loser_name"].apply(lambda x : x.lower().replace("-"," "))
        
    index= sub_data.index    
    columns = ['w_ace', 'l_ace', 'w_df', 'l_df', 'w_svpt',
               'l_svpt', 'w_1stIn', 'l_1stIn', 'w_1stWon', 'l_1stWon', 'w_2ndWon',
               'l_2ndWon', 'w_SvGms', 'l_SvGms', 'w_bpSaved', 'l_bpSaved', 'w_bpFaced',
               'l_bpFaced', 'minutes']
    
    sub_data["winner_name"] = sub_data["winner_name"].apply(lambda x : x.lower().replace("-"," "))
    sub_data["loser_name"] = sub_data["loser_name"].apply(lambda x : x.lower().replace("-"," "))
    merged_with_missing = pd.merge(sub_data.drop(columns,axis=1), missing_stats[columns + ['winner_name', 'loser_name', 'tourney_id']], on = ['tourney_id', 'winner_name', 'loser_name'], how= "left")
    merged_with_missing.index = index
    for col in columns:
        data_atp.loc[index, col] = merged_with_missing[col]
    
    return data_atp
    
if __name__ == "__main__":
    data_atp = pd.read_csv(r"C:\Users\User\Documents\tennis\data\clean_datasets\historical\matches_elo_V1.csv")
    data_atp["Date"] = pd.to_datetime(data_atp["Date"], format = "%Y-%m-%d")
    missing_stats = match_stats_main(data_atp, redo = True)
    
    