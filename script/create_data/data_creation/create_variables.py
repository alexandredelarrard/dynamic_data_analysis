# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 21:47:09 2018

@author: User
"""


import pandas as pd
import numpy as np
import re
import time
from datetime import timedelta
import glob
import joblib
import seaborn as sns

from create_data.data_creation.extract_data_atp import import_data_atp

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


def total_win_tb(x, i):
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



def update_match_num(data):
    
    def add_days(x):
        if x[2]<128:
            return x[0] + timedelta(days=x[1])
        else:
            return x[0] + timedelta(days=x[1]*2)
    
    #### reverse match num, 0 == final and correct it
    match_num = []
    for id_tourney in data["tourney_id"].unique():
        nliste= abs(data.loc[data["tourney_id"] == id_tourney, "match_num"].max() - data.loc[data["tourney_id"] == id_tourney, "match_num"])
        match_num += list(nliste+1)
    data["match_num"] = match_num 
    
    data["id_round"] = round(np.log(data["draw_size"]/data["match_num"]) / np.log(2), 0)
    data.loc[data["id_round"]<0,"id_round"]=0
    
    return data
    
    
def prep_data(data, verbose=0):
    
    t0 = time.time()
    dataset = data.copy()
    
    #### create score variables
    dataset["total_games"] = dataset["score"].apply(lambda x : extract_games_number(x))
    dataset['N_set']  = dataset['score'].apply(lambda x : count_sets(x))
    dataset['score2'] = dataset['score'].str.split(" ")
    
    for i in range(1,6):
        dataset["S%i"%i] = dataset['score2'].apply(lambda x : set_extract(x, i))
        for j, w_l in enumerate(["w", "l"]):
            dataset[w_l + "_S%i"%i] = dataset["S%i"%i].apply(lambda x : games_extract(x, j)).fillna(-1).astype(int)
            
    ##### flag wrong stats as missing stats for suppression in the creation variable part 
    ratio = np.where((dataset["l_svpt"]/dataset["total_games"] <= 2) | (dataset["l_svpt"]/dataset["total_games"]>=6), 1 ,0)
    dataset["missing_stats"] = np.where((pd.isnull(dataset["w_ace"]))|(dataset["w_SvGms"] == 0)|(dataset["l_SvGms"] == 0)|(ratio==1), 1, 0)
    
    dataset = dataset.loc[dataset["missing_stats"] !=1]
         
    ### dummify court
    dataset["indoor_flag"] = np.where(dataset["indoor_flag"] == "Outdoor", 0,1).astype(int)
    
    ### create hard indoor and hard outdoor as two different surfaces
    dataset["surface"] = dataset["surface"] + "_" + dataset["indoor_flag"].astype(str)
    dataset.loc[dataset["surface"] == "Carpet_0", "surface"] = "Carpet_1"
    dataset.loc[dataset["surface"] == "Clay_1", "surface"] = "Clay_0"

    ### dummify hand player
    dataset["winner_hand"] = np.where(dataset["winner_hand"] == "Right-Handed", 1, 0).astype(int)   
    dataset["loser_hand"] = np.where(dataset["loser_hand"] == "Right-Handed", 1, 0).astype(int)
    
    #### match num updated
    dataset = update_match_num(dataset)

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
    dataset["l_2nd_srv_ret_won"] = dataset["w_svpt"].astype(int) - dataset["w_1stIn"].astype(int) - dataset["w_2ndWon"].astype(int)
    dataset["l_bp_converted"]    = dataset["w_bpFaced"].astype(int) - dataset["w_bpSaved"].astype(int)
    dataset["l_total_srv_won"]   = dataset["l_1stWon"].astype(int) + dataset["l_2ndWon"].astype(int)
    dataset["l_total_ret_won"]   = dataset["l_1st_srv_ret_won"].astype(int) + dataset["l_2nd_srv_ret_won"].astype(int)
    
    #### handle serv games not close to 0.5
    dataset["w_SvGms"] = np.where((dataset["w_SvGms"]/dataset["total_games"] <= 0.4) | (dataset["w_SvGms"]/dataset["total_games"]>=0.6), dataset["total_games"]*0.5, dataset["w_SvGms"])
    dataset["l_SvGms"] = np.where((dataset["l_SvGms"]/dataset["total_games"] <= 0.4) | (dataset["l_SvGms"]/dataset["total_games"]>=0.6), dataset["total_games"]*0.5, dataset["l_SvGms"])
     
    ### tie breaks
    dataset["Nbr_tie-breaks"]   = dataset['score2'].apply(lambda x : len(re.findall('\((.*?)\)', str(x))))
    dataset["w_tie-breaks_won"] = dataset['score2'].apply(lambda x : win_tb(x))
    dataset["l_tie-breaks_won"] = dataset["Nbr_tie-breaks"] - dataset["w_tie-breaks_won"]
    dataset["total_tie_break_w"] = dataset['score2'].apply(lambda x : total_win_tb(x,0))/dataset["Nbr_tie-breaks"]
    dataset["total_tie_break_l"] = dataset['score2'].apply(lambda x : total_win_tb(x,1))/dataset["Nbr_tie-breaks"]
    
    ### is birthday
    dataset["DOB_w"] = pd.to_datetime(dataset["DOB_w"], format = "%Y-%m-%d") 
    dataset["DOB_l"] = pd.to_datetime(dataset["DOB_l"], format = "%Y-%m-%d") 
    
    dataset["w_birthday"] =  np.where((dataset["month"] == dataset["DOB_w"].dt.month), dataset["day_of_month"] - dataset["DOB_w"].dt.day, 31).astype(int)   
    dataset["l_birthday"] =  np.where((dataset["month"] == dataset["DOB_l"].dt.month), dataset["day_of_month"] - dataset["DOB_l"].dt.day, 31).astype(int)   
    
    ### if home country
    dataset["w_home"] = np.where(dataset["tourney_country"] == dataset["winner_ioc"],1,0)
    dataset["l_home"] = np.where(dataset["tourney_country"] == dataset["loser_ioc"],1,0)
    
    ### imc
    dataset["w_imc"] = dataset["Weight_w"] / (dataset["winner_ht"]/100)**2
    dataset["l_imc"] = dataset["Weight_l"] / (dataset["loser_ht"]/100)**2
    
    ### nbr proportion total points won
    dataset['w_total_pts_won'] = (dataset['w_2ndWon'].astype(float) + dataset['w_1stWon'].astype(float) + dataset['w_total_ret_won'].astype(float)) / (dataset["l_svpt"].astype(float) + dataset["w_svpt"].astype(float))
    dataset['l_total_pts_won'] = (dataset['l_2ndWon'].astype(float) + dataset['l_1stWon'].astype(float) + dataset['l_total_ret_won'].astype(float)) / (dataset["l_svpt"].astype(float) + dataset["w_svpt"].astype(float))

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
         
    ### predict  minutes for mvs
    if pd.isnull(dataset["minutes"]).sum()>0:
        index = (pd.isnull(dataset["minutes"]))|((dataset["minutes"]/dataset["total_games"]<2)|(dataset["minutes"]/dataset["total_games"]>12))
        
        models_minutes = glob.glob(r"C:\Users\User\Documents\tennis\models\stats_infering\minutes\*.dat")
        
        X = dataset.loc[index, ['N_set', 'day_of_month', 'day_of_year', 'draw_size', 'indoor_flag', 'l_S1', 'l_S2',
                                'l_S3', 'l_S4', 'l_S5', 'l_imc', 'loser_age', 'loser_hand', 'loser_ht', 'loser_rank', 
                                'loser_rank_points', 'month', 'round_F', 'round_QF', 'round_R128', 'round_R16', 'round_R32',
                                'round_R64', 'round_RR', 'round_SF', 'surface_Carpet_1', 'surface_Clay_0', 'surface_Grass_0',
                                'surface_Hard_0', 'surface_Hard_1', 'total_games', 'total_tie_break_l', 'total_tie_break_w', 'w_S1',
                                'w_S2', 'w_S3', 'w_S4', 'w_S5', 'w_imc', 'winner_age', 'winner_hand', 'winner_ht', 'winner_rank', 
                                'winner_rank_points', 'year']]
        
        dataset.loc[index, "minutes"] = 0
        for model in models_minutes:
            clf = joblib.load(model)
            dataset.loc[index, "minutes"] += np.exp(clf.predict(X))
            
        dataset.loc[index, "minutes"] = dataset.loc[index, "minutes"] /5.0
    
    if verbose ==1:
        dataset["missing"] = np.where(index, 1,0)
        sns.lmplot(x = "minutes", y = "total_games", data = dataset, hue = "missing")
        del dataset["missing"]
        
    dataset = dataset.drop(["S1", "S2","S3", "S4", "S5", 'score2', "missing_stats"], axis = 1)
     
    print("[{0}s] 6) Create additionnal variables ".format(time.time() - t0))
        
    return dataset


if __name__ == "__main__":
    import os
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"
    path = os.environ["DATA_PATH"]  + "/brute_info/historical/brute_info_atp/"
    data_atp = import_data_atp(path, redo = False)
    data = prep_data(data_atp, verbose =1)
