# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 21:47:09 2018

@author: User
"""


import pandas as pd
import numpy as np
import re
import time

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
        
        
def win_tb(x):
    count = 0
    for se in x:
        if "(" in se:
            if int(se.split("-")[0]) > int(re.sub(r'\([^)]*\)', '', se.split("-")[1])):
                count +=1
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
    
    
def prep_data(data):
    
    t0 = time.time()
    dataset = data.copy()
         
    ### dummify court
    dataset.loc[dataset["indoor_flag"] == "Outdoor", "indoor_flag"] = "0"
    dataset.loc[dataset["indoor_flag"] == "Indoor", "indoor_flag"]  = "1"
    dataset["indoor_flag"] = dataset["indoor_flag"].astype(int)
    
    ### dummify hand player
    dataset.loc[dataset["winner_hand"] =="Right-Handed", "winner_hand"] = "1"
    dataset.loc[dataset["winner_hand"] !="Right-Handed", "winner_hand"] = "0" ### left hand and ambidextre
    dataset["winner_hand"] = dataset["winner_hand"].astype(int)
    dataset.loc[dataset["loser_hand"] =="Right-Handed", "loser_hand"] = "1"
    dataset.loc[dataset["loser_hand"] !="Right-Handed", "loser_hand"] = "0" ### left hand and ambidextre
    dataset["loser_hand"] = dataset["loser_hand"].astype(int)
    
    #### date into days
    dataset["Date"] = pd.to_datetime(dataset["Date"], format = "%Y-%m-%d")
    
    dataset["day_week"] = dataset["Date"].dt.dayofweek
    dataset["month"] = dataset["Date"].dt.month
    dataset["year"] = dataset["Date"].dt.year
    dataset["week"] = dataset["Date"].dt.week
    dataset["day_of_year"] = dataset["Date"].dt.dayofyear
    dataset["day_of_month"] = dataset["Date"].dt.day
    
    #### return stats
    dataset["w_1st_srv_ret_won"] = dataset["l_1stIn"].astype(int) - dataset["l_1stWon"].astype(int)
    dataset["w_2nd_srv_ret_won"] = dataset["l_svpt"].astype(int) - dataset["l_1stIn"].astype(int) - dataset["l_2ndWon"].astype(int)
    dataset["w_bp_converted"]    = dataset["l_bpFaced"].astype(int) - dataset["l_bpSaved"].astype(int)
    dataset["w_total_srv_won"]   = dataset["w_1stWon"].astype(int) + dataset["w_2ndWon"].astype(int)
    dataset["w_total_ret_won"]   = dataset["w_1st_srv_ret_won"].astype(int) + dataset["w_2nd_srv_ret_won"].astype(int)
    
    dataset["l_1st_srv_ret_won"] = dataset["w_1stIn"].astype(int) - dataset["w_1stWon"].astype(int)
    dataset["l_2nd_srv_ret_won"] = dataset["w_svpt"].astype(int) - dataset["w_1stIn"] - dataset["w_2ndWon"].astype(int)
    dataset["l_bp_converted"]    = dataset["w_bpFaced"].astype(int) - dataset["w_bpSaved"].astype(int)
    dataset["l_total_srv_won"]   = dataset["l_1stWon"].astype(int) + dataset["l_2ndWon"].astype(int)
    dataset["l_total_ret_won"]   = dataset["l_1st_srv_ret_won"].astype(int) + dataset["l_2nd_srv_ret_won"].astype(int)
    
    #### create score variables
    dataset["total_games"] = dataset["score"].apply(lambda x : extract_games_number(x))
    dataset['N_set']  = dataset['score'].apply(lambda x : count_sets(x))
    dataset['score'] = dataset['score'].str.split(" ")
    dataset["S1"] = dataset['score'].apply(lambda x : set_extract(x, 1))
    dataset["S2"] = dataset['score'].apply(lambda x : set_extract(x, 2))
    dataset["S3"] = dataset['score'].apply(lambda x :set_extract(x, 3))
    dataset["S4"] = dataset['score'].apply(lambda x :set_extract(x,4))
    dataset["S5"] = dataset['score'].apply(lambda x :set_extract(x, 5))
    
    for i in range(1,6):
        for j, w_l in enumerate(["w", "l"]):
            dataset[w_l + "_S%i"%i] = dataset["S%i"%i].apply(lambda x : games_extract(x, j))
            
    ### correct/match minutes
    dataset.loc[(dataset["tourney_id"] == "2017-0308")&(dataset["winner_name"] == "Hyeon Chung")&(dataset["loser_name"] == "Martin Klizan"), "minutes"] = 135
    dataset.loc[(dataset["tourney_id"] == "2016-M001")&(dataset["winner_name"] == "Gilles Muller")&(dataset["loser_name"] == "Jeremy Chardy"), "minutes"] = 90
    
    dataset.loc[(dataset["minutes"] <20)&(~pd.isnull(dataset["w_S5"])), "minutes"] = int(dataset.loc[~pd.isnull(dataset["w_S5"]), "minutes"].median())
    dataset.loc[(dataset["minutes"] <20)&(~pd.isnull(dataset["w_S4"]))&(pd.isnull(dataset["w_S5"])), "minutes"] = int(dataset.loc[(pd.isnull(dataset["w_S5"]))&(~pd.isnull(dataset["w_S4"])), "minutes"].median())
    dataset.loc[(dataset["minutes"] <20)&(~pd.isnull(dataset["w_S3"]))&(pd.isnull(dataset["w_S4"])), "minutes"] = int(dataset.loc[(pd.isnull(dataset["w_S4"]))&(~pd.isnull(dataset["w_S3"])), "minutes"].median())
    
    ### tie breaks
    dataset["Nbr_tie-breaks"]   = dataset['score'].apply(lambda x : len(re.findall('\((.*?)\)', str(x))))
    dataset["w_tie-breaks_won"] = dataset['score'].apply(lambda x : win_tb(x))
    dataset["l_tie-breaks_won"] = dataset["Nbr_tie-breaks"] - dataset["w_tie-breaks_won"]
    
    ### is birthday
    dataset["DOB_w"] = pd.to_datetime(dataset["DOB_w"], format = "%Y-%m-%d") 
    dataset["DOB_l"] = pd.to_datetime(dataset["DOB_l"], format = "%Y-%m-%d") 
    
    dataset["w_birthday"] =0
    dataset.loc[(dataset["month"] == dataset["DOB_w"].dt.month)&(dataset["day_of_month"] == dataset["DOB_w"].dt.day) ,"w_birthday"] =1
    dataset["l_birthday"] =0
    dataset.loc[(dataset["month"] == dataset["DOB_l"].dt.month)&(dataset["day_of_month"] == dataset["DOB_l"].dt.day) ,"l_birthday"] =1
    
    ### if home country
    dataset["w_home"] = 0
    dataset.loc[dataset["tourney_country"] == dataset["winner_ioc"], "w_home"] = 1
    dataset["l_home"] = 0
    dataset.loc[dataset["tourney_country"] == dataset["loser_ioc"], "l_home"] = 1
    
    ### imc
    dataset["w_imc"] = dataset["Weight_w"] / (dataset["winner_ht"]/100)**2
    dataset["l_imc"] = dataset["Weight_l"] / (dataset["loser_ht"]/100)**2
    
    ### return normalization suppressing double fault not seen as successes
    dataset['w_1st_srv_ret_won'] = dataset['w_1st_srv_ret_won'].astype(float) / (dataset["l_1stIn"].astype(float))
    dataset['l_1st_srv_ret_won'] = dataset['l_1st_srv_ret_won'].astype(float) / (dataset["w_1stIn"].astype(float))
    
    dataset['w_2nd_srv_ret_won'] = dataset['w_2nd_srv_ret_won'].astype(float) / (dataset["w_2nd_srv_ret_won"].astype(float) + dataset["l_2ndWon"].astype(float))
    dataset['l_2nd_srv_ret_won'] = dataset['l_2nd_srv_ret_won'].astype(float) / (dataset["l_2nd_srv_ret_won"].astype(float) + dataset["w_2ndWon"].astype(float))
    
    dataset['w_total_ret_won'] = dataset['w_total_ret_won'].astype(float) / (dataset["l_svpt"].astype(float))
    dataset['l_total_ret_won'] = dataset['l_total_ret_won'].astype(float) / (dataset["w_svpt"].astype(float))
    
    ### normalize match statistics
    # serv normalization
    for col in ["w_ace", "w_df", 'w_1stIn', 'w_1stWon', 'w_2ndWon', "w_total_srv_won"]:
        dataset[col] = dataset[col].astype(float) / (dataset["w_svpt"].astype(float))
        
    for col in ["l_ace", "l_df", 'l_1stIn', 'l_1stWon', 'l_2ndWon', "l_total_srv_won"]:    
        dataset[col] = dataset[col].astype(float) / (dataset["l_svpt"].astype(float))
    
    ## bp normalization
    for col in ["w_bp_converted", "w_bpSaved", "w_bpFaced"]:    
        dataset[col] = dataset[col].astype(float) / (dataset["w_SvGms"].astype(float))
        
    for col in ["l_bp_converted", "l_bpSaved", "l_bpFaced"]:    
        dataset[col] = dataset[col].astype(float) / (dataset["l_SvGms"].astype(float))
        
    dataset = dataset.drop(["S1", "S2","S3", "S4", "S5", 'score'],axis = 1)
    
    print("[{0}s] 7) Create additionnal variables ".format(time.time() - t0))
        
    return dataset
