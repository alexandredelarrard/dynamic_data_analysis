# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 20:29:59 2018

@author: User
"""
import os
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import time 

tqdm.pandas(tqdm())

def href_match_h2h(path):
    
    liste_files = glob.glob(path)
    
    for i, file in enumerate(liste_files):
        if i ==0:
            data = pd.read_csv(file)
        else:
            data = pd.concat([data, pd.read_csv(file)],axis=0)
            
    return data


def post_processing_players():
    
    dd = href_match_h2h(r"C:\Users\User\Documents\tennis\data\brute_info\match_stats/h2h_match_stats/*.csv")
    dd["year"] = dd.iloc[:,1].apply(lambda x : x.replace("http://www.atpworldtour.com/en/scores/","").split("/")[0]).astype(int)
    dd["tourney_id"] = dd["year"].astype(str) + "-" + dd.iloc[:,1].apply(lambda x : x.replace("http://www.atpworldtour.com/en/scores/","").split("/")[1]).astype(str)
    dd = dd.sort_values("year").reset_index()
    
    del dd["index"]
    del dd["Unnamed: 0"]
    
    dd = dd.rename(columns = {"0":"url", "2": "sr_player1", "4": "sr_player2", 
                        "5": "aces_player1", "7" :"aces_player2", "8":"df_player1", "10":"df_player2", 
                        "11": "1serv_player1", "13": "1serv_player2", "14": "1serv_won_player1", "16": "1serv_won_player2",
                        "17": "2serv_won_player1", "19": "2serv_won_player2", "20" : "bp_saved_player1", "22" : "bp_saved_player2",
                        "23": "serv_games_player1", "25": "serv_games_player2", "27" : "rr_player1", "29" : "rr_player2",
                        "30": "1serv_return_won_player1", "32": "1serv_return_won_player2", "33": "2serv_return_won_player1", "35": "2serv_return_won_player2",
                        "36": "bp_converted_player1", "38": "bp_converted_player2", "39": "return_games_player1", "41": "return_games_player2",
                        "52": "player1", "53" : "score_player1", "54":"player2", "55": "score_player2", "56": "time"})
    
    return dd[["tourney_id", "year", "url","sr_player1", "sr_player2","aces_player1","aces_player2","df_player1","df_player2","1serv_player1","1serv_player2"
               ,"1serv_won_player1","1serv_won_player2","2serv_won_player1","2serv_won_player2","bp_saved_player1","bp_saved_player2"
               ,"serv_games_player1","serv_games_player2","rr_player1","rr_player2","1serv_return_won_player1","1serv_return_won_player2",
               "2serv_return_won_player1","2serv_return_won_player2","bp_converted_player1","bp_converted_player2","return_games_player1","return_games_player2",
               "player1","score_player1","player2","score_player2","time"]]


def match_proba(x, data):
    try:
        sub_data = data.loc[data["id_t"] == x[1]]   
        sub_data["counts"] = sub_data["key"].apply(lambda y : sum([1 for a in x if a in y]))
    
    except Exception:
        print(x)
        return -1
    
    if len(sub_data["counts"])>0:
        if sum((sub_data["counts"] == max(sub_data["counts"]))*1)>1:
            return -1
        elif sum((sub_data["counts"] == max(sub_data["counts"]))*1) == 1:
            return sub_data.loc[sub_data["counts"] == max(sub_data["counts"]), "ID"].values[0]
    else:
        return -1
    
def data_match(data_atp, data_orig):
    
    data_orig["ID"] = data_orig["key"].progress_apply(lambda x : match_proba(x, data_atp))
    return data_orig


def create_key(x):
    
    try:
        return [int(x[0].split("-")[0]), int(x[0].split("-")[1])] + x[1].lower().replace(".","").replace("'","").replace("-"," ").split(" ") + x[2].lower().replace(".","").replace("'","").replace("-"," ").split(" ")
    except Exception:
        return [int(x[0].split("-")[0]), x[0].split("-")[1]] + x[1].lower().replace(".","").replace("'","").replace("-"," ").split(" ") + x[2].lower().replace(".","").replace("'","").replace("-"," ").split(" ")


def create_stat(new_data_orig, dd):
    
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
    
    return missing_stats[["ATP_ID", 'w_ace', 'w_df', 'w_svpt', 'w_1stIn',
                           'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace',
                           'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms',
                           'l_bpSaved', 'l_bpFaced', 'minutes']]
    
    
def match_stats_main(data_atp, redo = False):
    
    if redo:
        dd = post_processing_players()
        dd["key"] =  dd[["year", "tourney_id","player1","player2"]].apply(lambda x : [x[0], int(x[1].split("-")[1])] + x[2].lower().replace(".","").replace("'","").replace("-"," ").split(" ") + x[3].lower().replace(".","").replace("'","").replace("-"," ").split(" "), axis=1)
        dd["ID"] = range(len(dd))
        dd["id_t"] = dd["key"].apply(lambda x : x[1])
        
        #### create data merge with main 
        data = data_atp.copy()
        data["key"] = data[["tourney_id","winner_name","loser_name"]].apply(lambda x : create_key(x), axis=1)
        data["id_t"] = data["key"].apply(lambda x : x[1])
        missing_value_data = data.loc[pd.isnull(data["minutes"])]
        
        ### match id closest infos
        t0 = time.time()
        new_data_orig = data_match(dd, missing_value_data)
        print(time.time() - t0)
        
        ### Create stats needed for total data 
        missing_stats = create_stat(new_data_orig, dd)
        missing_stats.to_csv(r"C:\Users\User\Documents\tennis\data\brute_info\historical\correct_missing_values\missing_match_stats.csv", index= False)
        
    else:
        missing_stats = pd.read_csv(r"C:\Users\User\Documents\tennis\data\brute_info\historical\correct_missing_values\missing_match_stats.csv")
    
    return missing_stats
    
#if __name__ == "__main__":
#    missing_stats = match_stats_main(data_atp, redo = False)
    
    