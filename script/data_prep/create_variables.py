# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 21:47:09 2018

@author: User
"""


import pandas as pd
import numpy as np
import re

def set_extract(x, taille):
    if len(x)>=taille:    
        return re.sub(r'\([^)]*\)', '', x[taille-1])
    else:
        return np.nan
    
def games_extract(x, w_l):
    
    try:
        if x != "RET" and x != "DEF" and pd.isnull(x) == False and x != "":
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
    
    
def prep_data(data):
    
    dataset = data.copy()
    
    ### dummify court
    dataset.loc[dataset["indoor_flag"] == "Outdoor", "indoor_flag"] = "0"
    dataset.loc[dataset["indoor_flag"] == "Indoor", "indoor_flag"]  = "1"
    dataset["indoor_flag"] = dataset["indoor_flag"].astype(int)
    
    #### date into days
    dataset["Date"] = pd.to_datetime(dataset["Date"], format = "%Y-%m-%d")
    
    dataset["day_week"] = dataset["Date"].dt.dayofweek
    dataset["month"] = dataset["Date"].dt.month
    dataset["year"] = dataset["Date"].dt.year
    dataset["week"] = dataset["Date"].dt.week
    dataset["day_of_year"] = dataset["Date"].dt.dayofyear
    dataset["day_of_month"] = dataset["Date"].dt.day
    
    #### return stats
    dataset["w_1st_srv_ret_won"] = dataset["l_1stIn"] - dataset["l_1stWon"]
    dataset["w_2nd_srv_ret_won"] = dataset["l_svpt"] - dataset["l_1stIn"] - dataset["l_2ndWon"]
    dataset["w_bp_converted"]    = dataset["l_bpFaced"] - dataset["l_bpSaved"]
    dataset["w_total_srv_won"]   = dataset["w_1stWon"] + dataset["w_2ndWon"]
    dataset["w_total_ret_won"]   = dataset["w_1st_srv_ret_won"] + dataset["w_2nd_srv_ret_won"]
    
    dataset["l_1st_srv_ret_won"] = dataset["w_1stIn"] - dataset["w_1stWon"]
    dataset["l_2nd_srv_ret_won"] = dataset["w_svpt"] - dataset["w_1stIn"] - dataset["w_2ndWon"]
    dataset["l_bp_converted"]    = dataset["w_bpFaced"] - dataset["w_bpSaved"]
    dataset["l_total_srv_won"]   = dataset["l_1stWon"] + dataset["l_2ndWon"]
    dataset["l_total_ret_won"]   = dataset["l_1st_srv_ret_won"] + dataset["l_2nd_srv_ret_won"]
    
    #### create score variables
    dataset['score'] = dataset['score'].str.split(" ")
    dataset["S1"] = dataset['score'].apply(lambda x : set_extract(x, 1))
    dataset["S2"] = dataset['score'].apply(lambda x : set_extract(x, 2))
    dataset["S3"] = dataset['score'].apply(lambda x :set_extract(x, 3))
    dataset["S4"] = dataset['score'].apply(lambda x :set_extract(x,4))
    dataset["S5"] = dataset['score'].apply(lambda x :set_extract(x, 5))
    
    for i in range(1,6):
        for j, w_l in enumerate(["w", "l"]):
            dataset[w_l + "_S%i"%i] = dataset["S%i"%i].apply(lambda x : games_extract(x, j))
    
    dataset["status"] = "Completed"
    index = dataset["score"].apply(lambda x : "RET" in x)
    dataset.loc[index, "status"] = "Retired"
    index = dataset["score"].apply(lambda x : "DEF" in x)
    dataset.loc[index, "status"] = "Def"
    
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
    
    
    ### imc
    dataset["w_imc"] = dataset["Weight_w"] / (dataset["winner_ht"]/100)**2
    dataset["l_imc"] = dataset["Weight_l"] / (dataset["loser_ht"]/100)**2
        
    return dataset

