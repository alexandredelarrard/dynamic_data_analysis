# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 13:38:19 2018

@author: JARD
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial
import time
import sys

sys.path.append(r"C:\Users\User\Documents\tennis\dynamic_data_analysis\script")
from create_data.utils.weight_past_matches import calculate_corr_surface, calculate_corr_time

def parallelize_dataframe(df, function, dictionnary, njobs):
    df_split = np.array_split(df, njobs)
    pool = Pool(njobs)
    func = partial(function, dictionnary)
    df2 = pd.concat(pool.map(func, df_split))
    
    pool.close()
    pool.join()
    
    return df2


def common_opponents(x, data):
    
    oppo_win  = data[np.where(data[:,1] == x[1]), 2].tolist()[0] +  data[np.where(data[:,2] == x[1]), 1].tolist()[0]
    oppo_lose = data[np.where(data[:,2] == x[2]), 1].tolist()[0] +  data[np.where(data[:,1] == x[2]), 2].tolist()[0]
    
    interset_ids = list(set.intersection(set(oppo_win), set(oppo_lose)))
    players = [x[1], x[2]]
    
    index= np.where(((np.isin(data[:,1], players)) & (np.isin(data[:,2], interset_ids + players))) |
                        ((np.isin(data[:,2], players)) & (np.isin(data[:,1], interset_ids + players))))
    sub_data = data[index]
    
    return sub_data


def add_weight(x, sub_data, corr_surface, corr_time):
    
    def diff_month(d1, d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month
    
    weight = list(diff_month(x[0], pd.to_datetime(sub_data[:,0])))
    weight = pd.DataFrame(sub_data[:,3])[0].map(corr_surface[x[3]])*pd.DataFrame(weight)[0].map(corr_time)
    weight = np.where(pd.isnull(weight), 0 , weight)
    weight = np.where(sub_data[:,5] ==1, weight*0.5, weight)
    b= np.concatenate((sub_data, np.expand_dims(weight, 1)), axis=1)

    return b


def weighted_statistics(x, liste_dataframe):
    
    data = liste_dataframe[0].copy()
    corr_surface = liste_dataframe[1]
    corr_time = liste_dataframe[2]
    
    #### calculate weight and stats if common opponents is not empty
    data_date = data[np.where((data[:,0] < x[0]))]
   
    if data_date.shape[0] > 0:
        sub_data = common_opponents(x, data_date)
        
        if sub_data.shape[0]>0:
            sub_data = add_weight(x, sub_data, corr_surface, corr_time)
            sub_data = sub_data[np.where(sub_data[:,-1] >0)]
            stats    = get_stats(x, sub_data)
        else:
            stats = [(0, )   + (np.nan,)*17]
    else:
        stats = [(0, )   + (np.nan,)*17]
    
    return stats


def get_stats(x, sub_data):
    
    """
    columns order :   'Date', 'winner_id', 'loser_id', "surface", 'minutes', 'missing_stats', "winner_rank", 'loser_rank', 'w_ace', 'w_df',  
                      'w_svpt','w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced','l_ace', 'l_df', 'l_svpt',
                      'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced','w_1st_srv_ret_won','w_2nd_srv_ret_won', 'w_bp_converted', 'w_total_srv_won',
                      'w_total_ret_won', 'l_1st_srv_ret_won', 'l_2nd_srv_ret_won', 'l_bp_converted','l_total_srv_won', 'l_total_ret_won', 'w_tie-breaks_won', 'l_tie-breaks_won', 'Nbr_tie-breaks', "N_set",
                      'l_total_pts_won', 'w_total_pts_won', "match_num"
                      
     x : "Date", "winner_id", "loser_id", "surface"
    """
    
    winner_w_data = sub_data[sub_data[:,1] == x[1]]
    winner_l_data = sub_data[sub_data[:,2] == x[1]]
    loser_w_data = sub_data[sub_data[:,1] == x[2]]
    loser_l_data = sub_data[sub_data[:,2] == x[2]]
    
    weight_winner = winner_w_data[:,43].sum() + winner_l_data[:,43].sum()
    weight_loser = loser_w_data[:,43].sum() + loser_l_data[:,43].sum()
    
    ws1 = (((winner_w_data[:,11]*winner_w_data[:,12] + (1-winner_w_data[:,11])*winner_w_data[:,13])*winner_w_data[:,43]).sum() 
             + ((winner_l_data[:,20]*winner_l_data[:,21] + (1-winner_l_data[:,20])*winner_l_data[:,22])*winner_l_data[:,43]).sum())/weight_winner 
    
    ws2 = (((loser_w_data[:,11]*loser_w_data[:,12] + (1-loser_w_data[:,11])*loser_w_data[:,13])*loser_w_data[:,43]).sum()
             + ((loser_l_data[:,20]*loser_l_data[:,21] + (1-loser_l_data[:,20])*loser_l_data[:,22])*loser_l_data[:,43]).sum())/weight_loser 
    
    wr1 = (((winner_w_data[:,26]*winner_w_data[:,20] + (1-winner_w_data[:,20])*winner_w_data[:,27])*winner_w_data[:,43]).sum() 
             + ((winner_l_data[:,31]*winner_l_data[:,11] + (1-winner_l_data[:,11])*winner_l_data[:,32])*winner_l_data[:,43]).sum())/weight_winner 
    
    wr2 = (((loser_w_data[:,26]*loser_w_data[:,20] + (1-loser_w_data[:,20])*loser_w_data[:,27])*loser_w_data[:,43]).sum() 
             + ((loser_l_data[:,31]*loser_l_data[:,11] + (1-loser_l_data[:,11])*loser_l_data[:,32])*loser_l_data[:,43]).sum())/weight_loser
    
    count = (sub_data.shape[0], ### confidence on stat
             
             ((winner_w_data[:,8]*winner_w_data[:,43]).sum()  + (winner_l_data[:,17]*winner_l_data[:,43]).sum())/weight_winner -\
             ((loser_w_data[:,8]*loser_w_data[:,43]).sum()  + (loser_l_data[:,17]*loser_l_data[:,43]).sum())/weight_loser, #### difference proportion aces
             
             ((winner_w_data[:,9]*winner_w_data[:,43]).sum()  + (winner_l_data[:,18]*winner_l_data[:,43]).sum())/weight_winner -\
             ((loser_w_data[:,9]*loser_w_data[:,43]).sum()  + (loser_l_data[:,18]*loser_l_data[:,43]).sum())/weight_loser, #### difference proportion df
             
             ((winner_w_data[:,11]*winner_w_data[:,43]).sum()  + (winner_l_data[:,20]*winner_l_data[:,43]).sum())/weight_winner -\
             ((loser_w_data[:,11]*loser_w_data[:,43]).sum()  + (loser_l_data[:,20]*loser_l_data[:,43]).sum())/weight_loser, #### difference proportion first serv
             
             ((winner_w_data[:,12]*winner_w_data[:,43]).sum()  + (winner_l_data[:,21]*winner_l_data[:,43]).sum())/weight_winner -\
             ((loser_w_data[:,12]*loser_w_data[:,43]).sum()  + (loser_l_data[:,21]*loser_l_data[:,43]).sum())/weight_loser, #### difference proportion first won
             
             ((winner_w_data[:,13]*winner_w_data[:,43]).sum()  + (winner_l_data[:,22]*winner_l_data[:,43]).sum())/weight_winner -\
             ((loser_w_data[:,13]*loser_w_data[:,43]).sum()  + (loser_l_data[:,22]*loser_l_data[:,43]).sum())/weight_loser, #### difference proportion second won
             
             #### overall skill on serv =  w1sp*fs + (1-fs)*w2sp
             ws1 - ws2, 
            
             ### overall skill on return = w1_ret* (l_fs) + (1- l_fs)*w2_ret
             wr1 - wr2, 
             
             ### overall skill on both
             ws1*ws1 - ws2*wr2,
             
             ### difference serv return 
             ws1 - wr2,
             
             ### difference serv return 2
             ws2- wr1,
             
             ### break point competencies  = bp_saved * bp_converted
             ((winner_w_data[:,15]*winner_w_data[:,28]*winner_w_data[:,43]).sum() + (winner_l_data[:,24]*winner_l_data[:,33]*winner_l_data[:,43]).sum())/weight_winner -\
             ((loser_w_data[:,15]*loser_w_data[:,28]*loser_w_data[:,43]).sum()  + (loser_l_data[:,24]*loser_l_data[:,33]*loser_l_data[:,43]).sum())/weight_loser, 
             
             ### tie break competencies 
             ((winner_w_data[:,36]*winner_w_data[:,43]/winner_w_data[:,39]).sum() + (winner_l_data[:,37]*winner_l_data[:,43]/winner_l_data[:,39]).sum())/weight_winner -  
             ((loser_w_data[:,36]*loser_w_data[:,43]/loser_w_data[:,39]).sum() + (loser_l_data[:,37]*loser_l_data[:,43]/loser_l_data[:,39]).sum())/weight_loser,
             
             ### proportion victory 1 vs 2 
             (sub_data[(sub_data[:,1] == x[1]) & (sub_data[:,2] == x[1])].shape[0] - 
             sub_data[(sub_data[:,1] == x[2]) & (sub_data[:,2] == x[2])].shape[0])/ sub_data.shape[0], 
             
              ### proportion victory common adversories
             (sub_data[(sub_data[:,1] == x[1])].shape[0] - 
              sub_data[(sub_data[:,1] == x[2])].shape[0])/ sub_data.shape[0],
             
             ### proportion points won common adversaries
             ((winner_w_data[:,41]*winner_w_data[:,43]).sum()  + (winner_l_data[:,40]*winner_l_data[:,43]).sum())/weight_winner -\
             ((loser_w_data[:,41]*loser_w_data[:,43]).sum()  + (loser_l_data[:,40]*loser_l_data[:,43]).sum())/weight_loser, #### difference proportion second won
             
             #### diff mean rank common adversaries
             ((winner_w_data[:,7]*winner_w_data[:,43]).sum()  + (winner_l_data[:,6]*winner_l_data[:,43]).sum())/weight_winner -\
             ((loser_w_data[:,7]*loser_w_data[:,43]).sum()  + (loser_l_data[:,6]*loser_l_data[:,43]).sum())/weight_loser, #### difference proportion second won
             
             ### diff weights
             weight_winner - weight_loser,
             
             )
    
    return [count]
    
def execute_stats(wrong_word_dict, data):
    count = data.apply(lambda x: weighted_statistics(x, wrong_word_dict))
    return count

def fatigue_games(x , data):
    """
    - x : "ref_days", "winner_id", "loser_id"
    - data: "ref_days", "winner_id", "loser_id", "total_games"
    """

    index_days = np.where(((x[0] - data[:,0]) >0)&((x[0] - data[:,0]) <=3))
    sub_data = data[index_days]
    
    index_1 = np.where(((sub_data[:,1] == x[1]) | (sub_data[:,2] == x[1])))
    index_2 = np.where(((sub_data[:,1] == x[2]) | (sub_data[:,2] == x[1])))
    
    ### number of games played during last days
    fatigue_w = sub_data[index_1, 3].sum()
    fatigue_l = sub_data[index_2, 3].sum()
    
    return fatigue_w - fatigue_l


def total_score(x):
    try:
        return sum([int(i.replace("[","").replace("]","").replace("'","")) for i in x if i !=""])
    except Exception:
        print(x)
        return 0


def global_stats(data):
    
    data = data.copy()
    data["diff_age"] = ((data["Date"] - data["DOB_w"]).dt.days - (data["Date"] - data["DOB_l"]).dt.days)/365
    data["diff_ht"] = data["winner_ht"] - data["loser_ht"]
    data["diff_imc"] = data["w_imc"] - data["l_imc"]
    data["diff_weight"] = data["Weight_w"] - data["Weight_l"]
    data["diff_year_turned_pro"] = data['Turned pro_w'] - data['Turned pro_l']
    
    #### normalize rank points and elo over time with a difference impact more important when pts and elo are low (concav impact)
    data["diff_elo"] = (data['elo1'] - data['elo2'])/np.log((data['elo1'] + data['elo2'])*0.5)
    data["diff_rank"] = data['winner_rank'] - data['loser_rank']
    data["diff_rk_pts"] = (data['winner_rank_points'] - data['loser_rank_points'])/np.log((data['winner_rank'] + data['loser_rank'])*0.5)
    data["diff_hand"] = data['winner_hand'] - data['loser_hand']
    data["diff_home"] = data['w_home'] - data['l_home']
    
    return data

def get_correlations(data, redo = False):
    
    if redo :
        #### calculate correlations
        data1 = data[["Date", "winner_id", "loser_id", "surface", "tourney_id"]].copy()
        data1["target"] = 1
        
        data2 = data1.copy()
        data2 = data2.rename(columns = {"winner_id" : "loser_id", "loser_id" : "winner_id"})
        data2["target"] = 0
        data2 = data2[["winner_id", "loser_id", "Date", "target", "surface", "tourney_id"]]
        
        tot = pd.concat([data1, data2], axis= 0)
        tot["Date"] = pd.to_datetime(tot["Date"], format = "%Y-%m-%d")
        
        correlation_surface   = calculate_corr_surface(tot, start_year=1990, end_year=2016, redo=redo)
        correlation_time      = calculate_corr_time(tot, start_year=1990, end_year=2016, redo=redo)
        
    else:    
        correlation_surface   = calculate_corr_surface(data, redo)
        correlation_time      = calculate_corr_time(data, redo)
        
    return correlation_surface, correlation_time


def create_stats(data, liste_dataframe):
    
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"], format = "%Y-%m-%d")
    data["DOB_w"] = pd.to_datetime(data["DOB_w"], format = "%Y-%m-%d")
    data["DOB_l"] = pd.to_datetime(data["DOB_l"], format = "%Y-%m-%d")
    
    #### get differrence of fatigue between players
    t0 = time.time()
    data["ref_days"]= (pd.to_datetime(data["Date"], format = "%Y-%m-%d")- pd.to_datetime("1901-01-01")).dt.days
    data["diff_fatigue_games"] = np.apply_along_axis(fatigue_games, 1, np.array(data[["ref_days", "winner_id", "loser_id"]]), np.array(data[["ref_days", "winner_id", "loser_id", "total_games"]]))
    del data["ref_days"]
    print("[{0:.2f}] Created diff fatigue games variables".format(time.time() - t0))
    
    data = global_stats(data)
    data["target"] = 1
    print(" Created target and global_stats variables ")
    
    #############################  calculate all necessary stats   ##########################################
    t0 = time.time()
    counts = np.apply_along_axis(weighted_statistics, 1, np.array(data[["Date", "winner_id", "loser_id", "surface"]]), liste_dataframe)
    counts = counts.reshape(counts.shape[0], counts.shape[2])
    
    ###### put the right name to the right column
    stats_cols = ["Common_matches", "diff_aces", "diff_df", "diff_1st_serv_in", "diff_1st_serv_won", "diff_2nd_serv_won",
                 "diff_skill_serv", "diff_skill_ret", "diff_overall_skill", "diff_serv1_ret2", "diff_serv2_ret1", "diff_bp", "diff_tie_break",
                 "diff_victories_12", "diff_victories_common_matches", "diff_pts_common_matches", "diff_mean_rank_adversaries", "diff_weights"]
    stats = pd.DataFrame(counts, columns = stats_cols)
    data = pd.concat([data, stats], axis = 1)

    print("exec stats {0:.2f}".format(time.time()-t0))
    
    ############################# create reverse data ##########################################
    ###### target
   
    data2 = data.copy()
    data2["target"] = 0
    
    for col in [x for x in data2.columns if "w_" == x[:2]]:
        data2.rename(columns={col : col.replace("w_","l_"), col.replace("w_","l_") : col}, inplace = True)
        
    for col in [x for x in data2.columns if "_w" == x[-2:]]:
        data2.rename(columns={col : col.replace("_w","_l"), col.replace("_w","_l") : col}, inplace = True)
    
    for col in [x for x in data2.columns if "winner_" in x]:
        data2.rename(columns={col : col.replace("winner_","loser_"), col.replace("winner_","loser_") : col}, inplace = True)
    
    data2.rename(columns={"elo1" : "elo2", "elo2" : "elo1"}, inplace = True)
    data2["prob_elo"] = 1 / (1 + 10 ** ((data2["elo2"] - data2["elo1"]) / 400))
            
    for col in [x for x in data2.columns if "diff_" == x[:5]]:
        data2[col] = -1*data2[col]
        
    total_data = pd.concat([data, data2], axis= 0)
    
    return total_data
    

def create_statistics(data, redo = False):
    
    #### get correlations coefficient
    correlation_surface, correlation_time = get_correlations(data, redo = redo)
        
    ############################# calculation of statistics ########################################## 
    calculate_stats = ['Date', 'winner_id', 'loser_id', "surface", 'minutes', 'missing_stats', "winner_rank", 'loser_rank', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
                     'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced','w_1st_srv_ret_won',
                     'w_2nd_srv_ret_won', 'w_bp_converted', 'w_total_srv_won', 'w_total_ret_won', 'l_1st_srv_ret_won', 'l_2nd_srv_ret_won', 'l_bp_converted',
                     'l_total_srv_won', 'l_total_ret_won', 'w_tie-breaks_won', 'l_tie-breaks_won', 'Nbr_tie-breaks', "N_set", 'l_total_pts_won', 'w_total_pts_won', "match_num"]

    liste_params = [np.array(data[calculate_stats]), correlation_surface, correlation_time]
    total_data = create_stats(data, liste_params)
    
    return total_data


if __name__ == "__main__": 
    import os
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"

    data = pd.read_csv(r"C:\Users\User\Documents\tennis\data\clean_datasets\historical\matches_elo_variables_V1.csv")
    data["Date"] = pd.to_datetime(data["Date"], format = "%Y-%m-%d")
    data["DOB_w"] = pd.to_datetime(data["DOB_w"], format = "%Y-%m-%d")
    data["DOB_l"] = pd.to_datetime(data["DOB_l"], format = "%Y-%m-%d")
   
    tot = create_statistics(data)
    