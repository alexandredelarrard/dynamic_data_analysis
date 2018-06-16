# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 13:38:19 2018

@author: JARD
"""

import pandas as pd
import numpy as np
import time

from create_train.utils.weight_past_matches import get_correlations

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
    b= np.concatenate((sub_data, np.expand_dims(weight, 1)), axis=1)

    return b


def convert(x):
    try:
        x = str(int(x))
    except Exception:
        x = str(x)
        
    return x


def weighted_statistics(x, liste_dataframe):
    
    data = liste_dataframe[0].copy()
    corr_surface = liste_dataframe[1]
    corr_time = liste_dataframe[2]
    
    #### calculate weight and stats if common opponents is not empty
    data_date = data[np.where((data[:,0] < x[0]))]
   
    if data_date.shape[0] > 0:
        
        sub_data = add_weight(x, data_date, corr_surface, corr_time)
        sub_data = sub_data[np.where(sub_data[:,-1] >0)]
        
        #### each players stat on historic only
        if sub_data.shape[0]>0:
            stats1 = get_player_stat(x, sub_data)
        else:
            stats1 = (np.nan,)*8

        #### second part on common opponents
        sub_data = common_opponents(x, sub_data)
        
        if sub_data.shape[0]>0:
            stats    = get_stats(x, sub_data)
            stats = [(stats) + (stats1)]
        else:
            stats = [(0,)   + (np.nan,)*18+ (stats1)]
    else:
        stats = [(0, )   + (np.nan,)*26]
        print(x)

    return stats


def get_player_stat(x, sub_data):
    
    w_index= (sub_data[:,1] == x[1])|(sub_data[:,2] == x[1])
    l_index = (sub_data[:,1] == x[2])|(sub_data[:,2] == x[2])
    
    if sub_data[w_index].shape[0] > 0:
        best_rank_winner = np.min(sub_data[w_index, 6])
    else:
        best_rank_winner = -1
    
    if sub_data[l_index].shape[0] > 0:
        best_rank_loser  = np.min(sub_data[l_index, 7])
    else:
        best_rank_loser = -1

    try:
        prop_victory_surface_winner = sub_data[(sub_data[:,1] == x[1])&(sub_data[:,3] == x[3])].shape[0]/sub_data[(w_index)&(sub_data[:,3] == x[3])].shape[0]
    except Exception:
        prop_victory_surface_winner = -1
        pass
    
    try:
        prop_victory_surface_loser  = sub_data[(sub_data[:,2] == x[2])&(sub_data[:,3] == x[3])].shape[0]/sub_data[(l_index)&(sub_data[:,3] == x[3])].shape[0]
    except Exception:
        prop_victory_surface_loser = -1
        pass
        
#    delta_rank_1mois_winner = 
#    delta_rank_1mois_loser  = 
    
#    jeu_consecutif_10_match_w = 
#    jeu_consecutif_10_match_l = 
    
    ### 39 = N_set  6 = best_of 
    try:
        prop_last_set_gagne_w =  sub_data[(sub_data[:,1] == x[1])&(sub_data[:,39] == x[6])].shape[0] / sub_data[(w_index)&(sub_data[:,39] == x[6])].shape[0]
    except Exception:
        prop_last_set_gagne_w = -1
        pass
    
    try:
        prop_last_set_gagne_l =  sub_data[(sub_data[:,1] == x[2])&(sub_data[:,39] == x[6])].shape[0] / sub_data[(l_index)&(sub_data[:,39] == x[6])].shape[0] 
    except Exception:
        prop_last_set_gagne_l = -1
        pass
        
    nbr_same_level_trns_w = sub_data[(w_index)&(sub_data[:,42] == x[4])&(x[5]>=sub_data[:,43])].shape[0]
    nbr_same_level_trns_l = sub_data[(l_index)&(sub_data[:,42] == x[4])&(x[5]>=sub_data[:,43])].shape[0]  

    return (best_rank_winner, best_rank_loser, prop_victory_surface_winner, prop_victory_surface_loser,
            nbr_same_level_trns_w, nbr_same_level_trns_l, prop_last_set_gagne_w, prop_last_set_gagne_l)


def get_stats(x, sub_data):
    
    """
    columns order :   'Date', 'winner_id', 'loser_id', "surface", 'minutes', 'best_of', "winner_rank", 'loser_rank', 'w_ace', 'w_df',  
                      'w_svpt','w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced','l_ace', 'l_df', 'l_svpt',
                      'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced','w_1st_srv_ret_won','w_2nd_srv_ret_won', 'w_bp_converted', 'w_total_srv_won',
                      'w_total_ret_won', 'l_1st_srv_ret_won', 'l_2nd_srv_ret_won', 'l_bp_converted','l_total_srv_won', 'l_total_ret_won', 'w_tie-breaks_won', 'l_tie-breaks_won', 'Nbr_tie-breaks', "N_set",
                      'l_total_pts_won', 'w_total_pts_won', 'tourney_id', "round", 'weight'
                      
     x : "Date", "winner_id", "loser_id", "surface", "tourney_id_wo_year", "round", 'best_of'
    """
    
    winner_w_data = sub_data[sub_data[:,1] == x[1]]
    winner_l_data = sub_data[sub_data[:,2] == x[1]]
    loser_w_data = sub_data[sub_data[:,1] == x[2]]
    loser_l_data = sub_data[sub_data[:,2] == x[2]]
    
    weight_winner = winner_w_data[:,-1].sum() + winner_l_data[:,-1].sum()
    weight_loser = loser_w_data[:,-1].sum() + loser_l_data[:,-1].sum()
    
    ws1 = (((winner_w_data[:,11]*winner_w_data[:,12] + (1-winner_w_data[:,11])*winner_w_data[:,13])*winner_w_data[:,-1]).sum() 
             + ((winner_l_data[:,20]*winner_l_data[:,21] + (1-winner_l_data[:,20])*winner_l_data[:,22])*winner_l_data[:,-1]).sum())/weight_winner 
    
    ws2 = (((loser_w_data[:,11]*loser_w_data[:,12] + (1-loser_w_data[:,11])*loser_w_data[:,13])*loser_w_data[:,-1]).sum()
             + ((loser_l_data[:,20]*loser_l_data[:,21] + (1-loser_l_data[:,20])*loser_l_data[:,22])*loser_l_data[:,-1]).sum())/weight_loser 
    
    wr1 = (((winner_w_data[:,26]*winner_w_data[:,20] + (1-winner_w_data[:,20])*winner_w_data[:,27])*winner_w_data[:,-1]).sum() 
             + ((winner_l_data[:,31]*winner_l_data[:,11] + (1-winner_l_data[:,11])*winner_l_data[:,32])*winner_l_data[:,-1]).sum())/weight_winner 
    
    wr2 = (((loser_w_data[:,26]*loser_w_data[:,20] + (1-loser_w_data[:,20])*loser_w_data[:,27])*loser_w_data[:,-1]).sum() 
             + ((loser_l_data[:,31]*loser_l_data[:,11] + (1-loser_l_data[:,11])*loser_l_data[:,32])*loser_l_data[:,-1]).sum())/weight_loser
    
    count = (sub_data.shape[0], ### confidence on stat
             
             ((winner_w_data[:,8]*winner_w_data[:,-1]).sum()  + (winner_l_data[:,17]*winner_l_data[:,-1]).sum())/weight_winner -\
             ((loser_w_data[:,8]*loser_w_data[:,-1]).sum()  + (loser_l_data[:,17]*loser_l_data[:,-1]).sum())/weight_loser, #### difference proportion aces
             
             ((winner_w_data[:,9]*winner_w_data[:,-1]).sum()  + (winner_l_data[:,18]*winner_l_data[:,-1]).sum())/weight_winner -\
             ((loser_w_data[:,9]*loser_w_data[:,-1]).sum()  + (loser_l_data[:,18]*loser_l_data[:,-1]).sum())/weight_loser, #### difference proportion df
             
             ((winner_w_data[:,11]*winner_w_data[:,-1]).sum()  + (winner_l_data[:,20]*winner_l_data[:,-1]).sum())/weight_winner -\
             ((loser_w_data[:,11]*loser_w_data[:,-1]).sum()  + (loser_l_data[:,20]*loser_l_data[:,-1]).sum())/weight_loser, #### difference proportion first serv
             
             ((winner_w_data[:,12]*winner_w_data[:,-1]).sum()  + (winner_l_data[:,21]*winner_l_data[:,-1]).sum())/weight_winner -\
             ((loser_w_data[:,12]*loser_w_data[:,-1]).sum()  + (loser_l_data[:,21]*loser_l_data[:,-1]).sum())/weight_loser, #### difference proportion first won
             
             ((winner_w_data[:,13]*winner_w_data[:,-1]).sum()  + (winner_l_data[:,22]*winner_l_data[:,-1]).sum())/weight_winner -\
             ((loser_w_data[:,13]*loser_w_data[:,-1]).sum()  + (loser_l_data[:,22]*loser_l_data[:,-1]).sum())/weight_loser, #### difference proportion second won
             
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
             ((winner_w_data[:,15]*winner_w_data[:,28]*winner_w_data[:,-1]).sum() + (winner_l_data[:,24]*winner_l_data[:,33]*winner_l_data[:,-1]).sum())/weight_winner -\
             ((loser_w_data[:,15]*loser_w_data[:,28]*loser_w_data[:,-1]).sum()  + (loser_l_data[:,24]*loser_l_data[:,33]*loser_l_data[:,-1]).sum())/weight_loser, 
             
             ### tie break competencies 
             ((winner_w_data[:,36]*winner_w_data[:,-1]/winner_w_data[:,39]).sum() + (winner_l_data[:,37]*winner_l_data[:,-1]/winner_l_data[:,39]).sum())/weight_winner -  
             ((loser_w_data[:,36]*loser_w_data[:,-1]/loser_w_data[:,39]).sum() + (loser_l_data[:,37]*loser_l_data[:,-1]/loser_l_data[:,39]).sum())/weight_loser,
             
             ### proportion victory 1 vs 2 
             (sub_data[(sub_data[:,1] == x[1]) & (sub_data[:,2] == x[2])].shape[0] - 
              sub_data[(sub_data[:,1] == x[2]) & (sub_data[:,2] == x[1])].shape[0]) / sub_data[((sub_data[:,1] == x[1])&(sub_data[:,2] == x[2])) | ((sub_data[:,1] == x[2])&(sub_data[:,2] == x[1]))].shape[0], 
             
              ### proportion victory common adversories
             (sub_data[(sub_data[:,1] == x[1])].shape[0] - 
              sub_data[(sub_data[:,1] == x[2])].shape[0])/ sub_data.shape[0],
             
             ### proportion points won common adversaries
             ((winner_w_data[:,41]*winner_w_data[:,-1]).sum()  + (winner_l_data[:,40]*winner_l_data[:,-1]).sum())/weight_winner -\
             ((loser_w_data[:,41]*loser_w_data[:,-1]).sum()  + (loser_l_data[:,40]*loser_l_data[:,-1]).sum())/weight_loser, #### difference proportion second won
             
             #### diff mean rank common adversaries
             ((winner_w_data[:,7]*winner_w_data[:,-1]).sum()  + (winner_l_data[:,6]*winner_l_data[:,-1]).sum())/weight_winner -\
             ((loser_w_data[:,7]*loser_w_data[:,-1]).sum()  + (loser_l_data[:,6]*loser_l_data[:,-1]).sum())/weight_loser, #### difference proportion second won
             
             ### diff weights
             weight_winner - weight_loser,
             
             ### diff average time per set
             ((winner_w_data[:,4]*winner_w_data[:,-1]).sum()  + (winner_l_data[:,4]*winner_l_data[:,-1]).sum())/((sub_data[(sub_data[:,1] == x[1]) | (sub_data[:,2] == x[1]), 39].sum())*weight_winner) -\
             ((loser_w_data[:,4]*loser_w_data[:,-1]).sum()  + (loser_l_data[:,4]*loser_l_data[:,-1]).sum())/((sub_data[(sub_data[:,1] == x[2]) | (sub_data[:,2] == x[2]), 39].sum())*weight_loser)
             
             #    prop_1er_set_gagne_surface_w = 
             #    prop_1er_set_gagne_surface_l =   
             
             #    prop_2eme_set_gagne_surface_w = 
             #    prop_2eme_set_gagne_surface_l =   
             
             #    prop_3eme_set_gagne_surface_w = 
             #    prop_3eme_set_gagne_surface_l =   
             
             )
    
    return count


def fatigue_games(x , data):
    """
    - x : "ref_days", "winner_id", "loser_id"
    - data: "ref_days", "winner_id", "loser_id", "total_games"
    """

    index_days = np.where((data[:,0] < x[0])&((x[0] - data[:,0]) <=3))
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

def correlation_subset_data(data, redo):
    
    #### get correlations coefficient
    correlation_surface, correlation_time = get_correlations(data, redo = redo)
    ############################# calculation of statistics ########################################## 
    ### we use tourney date instead of date because wont have last statistics of tourney match, atp crawl them and give them later
    cols_stat = ['Date', 'winner_id', 'loser_id', "surface", 'minutes', 'best_of', "winner_rank", 'loser_rank', 
                       'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
                       'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced',
                       'w_1st_srv_ret_won','w_2nd_srv_ret_won', 'w_bp_converted', 'w_total_srv_won', 'w_total_ret_won', 
                       'l_1st_srv_ret_won', 'l_2nd_srv_ret_won', 'l_bp_converted', 'l_total_srv_won', 'l_total_ret_won',
                       'w_tie-breaks_won', 'l_tie-breaks_won', 'Nbr_tie-breaks', "N_set", 'l_total_pts_won', 'w_total_pts_won',
                       'tourney_id_wo_year', "round", "total_games"]
    
    return cols_stat, correlation_surface, correlation_time

def global_stats(data):
    
    data = data.copy()
    data["diff_age"] = ((data["Date"] - data["DOB_w"]).dt.days - (data["Date"] - data["DOB_l"]).dt.days)/365
    data["diff_ht"] = data["winner_ht"] - data["loser_ht"]
    data["diff_imc"] = data["w_imc"] - data["l_imc"]
    data["diff_weight"] = data["Weight_w"] - data["Weight_l"]
    data["diff_year_turned_pro"] = data['Turned pro_w'] - data['Turned pro_l']
    
    #### normalize rank points and elo over time with a difference impact more important when pts and elo are low (concav impact)
    data["diff_elo"] = data['elo1'] - data['elo2']
    data["diff_rank"] = data['winner_rank'] - data['loser_rank']
    data["diff_rk_pts"] = (data['winner_rank_points'] - data['loser_rank_points'])/np.log((data['winner_rank'] + data['loser_rank'])*0.5)
    data["diff_hand"] = data['winner_hand'] - data['loser_hand']
    data["diff_home"] = data['w_home'] - data['l_home']
    
    return data


def create_stats(data, liste_params):
    
    mvs = pd.isnull(data).sum()
    print(mvs)

    data["Date"] = pd.to_datetime(data["Date"], format = "%Y-%m-%d")
    data["DOB_w"] = pd.to_datetime(data["DOB_w"], format = "%Y-%m-%d")
    data["DOB_l"] = pd.to_datetime(data["DOB_l"], format = "%Y-%m-%d")
    data["prob_elo"] = 1 / (1 + 10 ** ((data["elo2"] - data["elo1"]) / 400))
    
    #### get differrence of fatigue between players
    t0 = time.time()
    liste_params[0]["ref_days"]= (pd.to_datetime(liste_params[0]["Date"], format = "%Y-%m-%d")- pd.to_datetime("1901-01-01")).dt.days
    data["ref_days"]= (pd.to_datetime(data["Date"], format = "%Y-%m-%d")- pd.to_datetime("1901-01-01")).dt.days
    data["diff_fatigue_games"] = np.apply_along_axis(fatigue_games, 1, np.array(data[["ref_days", "winner_id", "loser_id"]]), np.array(liste_params[0][["ref_days", "winner_id", "loser_id", "total_games"]]))
    del data["ref_days"]
    liste_params[0] = liste_params[0].drop(["total_games", "ref_days"],axis=1)

    print("[{0:.2f}] Created diff fatigue games variables".format(time.time() - t0))
    
    data = global_stats(data)
    data["target"] = 1
    data = data.reset_index(drop=True)
    print(" Created target and global_stats variables ")
    
    #############################  calculate all necessary stats   ##########################################
    t0 = time.time()
    liste_params[0] = np.array(liste_params[0])
    counts = np.apply_along_axis(weighted_statistics, 1, np.array(data[["Date", "winner_id", "loser_id", "surface", "tourney_id_wo_year", "round", 'best_of']]), liste_params)
    counts = counts.reshape(counts.shape[0], counts.shape[2])
    
    ###### put the right name to the right column
    stats_cols = ["Common_matches", "diff_aces", "diff_df", "diff_1st_serv_in", "diff_1st_serv_won", "diff_2nd_serv_won",
                 "diff_skill_serv", "diff_skill_ret", "diff_overall_skill", "diff_serv1_ret2", "diff_serv2_ret1", "diff_bp", "diff_tie_break",
                 "diff_victories_12", "diff_victories_common_matches", "diff_pts_common_matches", "diff_mean_rank_adversaries", "diff_weights",
                 "diff_time_set",
                 "bst_rk_w", "bst_rk_l", "prop_victory_surface_w", "prop_victory_surface_l",  "nbr_reach_level_tourney_w", 
                 "nbr_reach_level_tourney_l", "prop_last_set_gagne_w", "prop_last_set_gagne_l"]
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
    
    data["tourney_id_wo_year"] =   list(list(zip(*data["tourney_id"].str.split("-")))[1])
    data["tourney_id_wo_year"] = "_" + data["tourney_id_wo_year"]
  
    cols_stat, correlation_surface, correlation_time = correlation_subset_data(data, redo)

    liste_params = [data[cols_stat], correlation_surface, correlation_time]
    total_data = create_stats(data, liste_params)
    
    return total_data


if __name__ == "__main__": 
    import os
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"

    data = pd.read_csv(r"C:\Users\User\Documents\tennis\data\clean_datasets\historical\matches_elo_variables_V1.csv")
    data["Date"] = pd.to_datetime(data["Date"], format = "%Y-%m-%d")
    data["tourney_date"] = pd.to_datetime(data["tourney_date"], format = "%Y-%m-%d")
    data["DOB_w"] = pd.to_datetime(data["DOB_w"], format = "%Y-%m-%d")
    data["DOB_l"] = pd.to_datetime(data["DOB_l"], format = "%Y-%m-%d")
    data = data.sort_values(["Date", "tourney_id"])
    
    tot = create_statistics(data)