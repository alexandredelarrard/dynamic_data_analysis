# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 10:29:30 2018

@author: User
"""

import pandas as pd
import glob
import time 
import numpy as np
import os
import re
import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (11,11)
sns.set(rc={'figure.figsize':(11,11)})

from create_data.utils.date_creation import deduce_match_date
from create_data.data_creation.extract_players import  merge_atp_players
from create_data.data_creation.merge_tourney  import merge_tourney


def set_extract(x, taille):
    if len(x)>=taille:    
        return re.sub(r'\([^)]*\)', '', x[taille-1])
    else:
        return np.nan
    
    
def games_extract(x, w_l):
    
    try:
        if x != "RET" and x != "W/O" and x != "W/O " and x != "DEF" and pd.isnull(x) == False and x != "":
            return x.split("-")[w_l]
        else:
            return np.nan
    except Exception:
        print(x)
        
        
def win_tb(x, i):
    count = 0
    for se in x:
        if "(" in se:
            if int(se.split("-")[0]) > int(re.sub(r'\([^)]*\)', '', se.split("-")[1])):
                if i ==1:
                    count += int(re.search(r'\((.*?)\)', se).group(1))
                else:
                    count += 2 + int(re.search(r'\((.*?)\)', se).group(1))
            else:
                if i ==0:
                    count += int(re.search(r'\((.*?)\)', se).group(1))
                else:
                    count += 2 + int(re.search(r'\((.*?)\)', se).group(1))
                    
    return count


def extract_games_number(x):
    try:
        x = re.sub(r'\([^)]*\)', '', x)
        x = x.replace(" ",",").replace("-",",").split(",")
        return sum([int(a) for a in x if a !=""])
    
    except Exception:
        print(x)
        
        
def count_sets(x):
    x = re.sub(r'\([^)]*\)', '', x)
    return x.count("-")


def import_data_atp(path):
    
    t0 = time.time()
    liste_files = glob.glob(path + "/*.csv")
    
    for i, file in enumerate(liste_files):
        if i == 0:
            data = pd.read_csv(file)
        else:
            data = pd.concat([data, pd.read_csv(file, encoding = "latin1")], axis=0)
            
    data = data.sort_values(["tourney_date", "tourney_id", "match_num"])

    # =============================================================================
    #     #### merge with tourney database 
    # =============================================================================
    data["tourney_date"]   = pd.to_datetime(data["tourney_date"], format = "%Y%m%d")  
    t0 = time.time()
    data = merge_tourney(data)
    print("[{0}s] 0) Merge with tourney database ".format(time.time() - t0))
    
    # =============================================================================
    #     #### deduce the date
    # =============================================================================
    data["Date"]  = data[["tourney_date", "tourney_end_date", "match_num", "draw_size", "round"]].apply(lambda x : deduce_match_date(x), axis=1)
    print("\n [{0}s] 1) Calculate date of match ".format(time.time() - t0))
    
    # =============================================================================
    #     #### correct little error
    # =============================================================================
    data = data.sort_values(["Date", "tourney_name"])
    
    ### identify a game as retired:  Andreas Seppi  Mario Ancic 2007-02-12    Marseille
    data.loc[(data["Date"] == "2007-02-12")&(data["winner_name"] == "Andreas Seppi")&(data["loser_name"] == "Mario Ancic"), "score"] = "RET"
    
    #### suppress davis cup and JO
    sp = data.shape[0] 
    not_davis_index = data["tourney_name"].apply(lambda x : "Davis Cup" not in x and "Olympic" not in x)
    data = data.loc[not_davis_index]
    print(" --- Suppress davis cup and JO : {0} ".format(sp - data.shape[0]))
    
    #### suppress challenger matches
    sp = data.shape[0] 
    data = data.loc[data["tourney_level"] != "C"]
    print(" --- Suppress challenger matches : {0} ".format(sp - data.shape[0]))
    
    ### create status of match
    data["status"] = "Completed"
    index = data["score"].apply(lambda x : "RET" in str(x))
    data.loc[index, "status"] = "Retired"
    index = data["score"].apply(lambda x : "DEF" in str(x) or "Jun" in str(x))
    data.loc[index, "status"] = "Def"
    index = data["score"].apply(lambda x : "W/O" in str(x) or "W/O " in str(x) or " W/O" in str(x))
    data.loc[index, "status"] = "Walkover"
    
    ### suppress not completed match
    data = data.loc[~data["status"].isin(["Retired", "Walkover", "Def"])]
    
    #### fill in missing scores
    data.loc[(data["tourney_id"] == "2007-533")&(pd.isnull(data["score"])), "score"] = "6-1 4-6 7-5"
    data.loc[(data["tourney_id"] == "1997-319")&(pd.isnull(data["score"])), "score"] = "6-4 6-4 6-4"
    
    data.loc[data["winner_name"] == "joshua goodall", "winner_name"] = "josh goodall"
    data.loc[data["loser_name"] == "joshua goodall", "loser_name"] = "josh goodall"
    data = data.reset_index(drop=True)
    print("\n [{0}s] 2) Import ATP dataset ".format(time.time() - t0))
    
    # =============================================================================
    #     #### merge players dataset to atp history
    # =============================================================================
    t0 = time.time()
    data = merge_atp_players(data)
    data = data.drop(["Players_ID_w", "Player_Name_w","Players_ID_l", "Player_Name_l"], axis=1)
    print("[{0}s] 3) Merge with players and fillin missing values".format(time.time() - t0))
                
    # =============================================================================
    #     #### create score variables
    # =============================================================================
    data["total_games"] = data["score"].apply(lambda x : extract_games_number(x))
    data['N_set']  = data['score'].apply(lambda x : count_sets(x))
    data['score2'] = data['score'].str.split(" ")
    for i in range(1,6):
        data["S%i"%i] = data['score2'].apply(lambda x : set_extract(x, i))
        for j, w_l in enumerate(["w", "l"]):
            data[w_l + "_S%i"%i] = data["S%i"%i].apply(lambda x : games_extract(x, j)).fillna(-1).astype(int)
            
    data["total_tie_break_w"]   = data['score2'].apply(lambda x : win_tb(x,0))
    data["total_tie_break_l"]   = data['score2'].apply(lambda x : win_tb(x,1))
    del data['score2']
    
    data.loc[(data["tourney_id"] == "2017-0308")&(data["winner_name"] == "Hyeon Chung")&(data["loser_name"] == "Martin Klizan"), "minutes"] = 135
    data.loc[(data["tourney_id"] == "2016-M001")&(data["winner_name"] == "Gilles Muller")&(data["loser_name"] == "Jeremy Chardy"), "minutes"] = 90
    
    data["indoor_flag"] = np.where(data["indoor_flag"] == "Outdoor", 0,1).astype(int)
    data["Date"] = pd.to_datetime(data["Date"], format = "%Y-%m-%d")

    data["month"] = data["Date"].dt.month
    data["year"] = data["Date"].dt.year
    data["day_of_year"] = data["Date"].dt.dayofyear
    data["day_of_month"] = data["Date"].dt.day
    
    ### create hard indoor and hard outdoor as two different surfaces
    data["surface"] = data["surface"] + "_" + data["indoor_flag"].astype(str)
    data.loc[data["surface"] == "Carpet_0", "surface"] = "Carpet_1"
    data.loc[data["surface"] == "Clay_1", "surface"] = "Clay_0"
    
    
    ### imc
    data["w_imc"] = data["Weight_w"] / (data["winner_ht"]/100)**2
    data["l_imc"] = data["Weight_l"] / (data["loser_ht"]/100)**2
    
    data =  data.drop(["Weight_w", "Weight_l", "masters", "S1", "S2","S3", "S4", "S5", 'tourney_id', 'tourney_name', 'tourney_level', 'tourney_date', 'match_num', 'winner_id', 'winner_seed', 
                      'winner_entry', 'winner_name',  'winner_ioc','loser_id', 'loser_seed', 'loser_entry', 'loser_name', 
                      'loser_ioc',  'Currency',  'tourney_country', 'tourney_city', 'DOB_w',  'DOB_l', 'prize', 'status'], axis=1)
    
    data["winner_hand"] = np.where(data["winner_hand"] == "Right-Handed", 1, 0).astype(int)   
    data["loser_hand"] = np.where(data["loser_hand"] == "Right-Handed", 1, 0).astype(int)
    
    cols_object = [x for x in data.columns if data[x].dtypes == "O"]
    for col in cols_object:
        if len(data[col].unique())< 15 :
            a = pd.get_dummies(data[col], prefix = col)
            data = pd.concat([a, data], axis = 1)
            del data[col]
        else:
            del data[col]
            
    print("[{0}s] 4) Create score and useful variables".format(time.time() - t0))
    
    test = data.loc[(pd.isnull(data["w_ace"]))|(data["w_SvGms"] == 0)|(data["l_SvGms"] == 0)]
    train = data[~(pd.isnull(data["w_ace"]))|(data["w_SvGms"] == 0)|(data["l_SvGms"] == 0)]
    
    test_minutes = data.loc[(pd.isnull(data["minutes"]))|((data["minutes"]/data["total_games"]<2)|(data["minutes"]/data["total_games"]>12))]
    train_minutes =data.loc[~(pd.isnull(data["minutes"]))|((data["minutes"]/data["total_games"]<2)|(data["minutes"]/data["total_games"]>12))]
    
    return train, test, train_minutes, test_minutes


def data_analysis(data):
    data["time/games"] = data["minutes"]/data["total_games"]
    data["time/games_01"] = np.where((data["time/games"]<2)|(data["time/games"]>12),1,0)
    sns.lmplot(x="total_games", y="minutes", hue="time/games_01", data=data)

    
if __name__ == "__main__":
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"
    path = os.environ["DATA_PATH"]  + "/brute_info/historical/brute_info_atp/"
    train, test, train_minutes, test_minutes = import_data_atp(path)

    