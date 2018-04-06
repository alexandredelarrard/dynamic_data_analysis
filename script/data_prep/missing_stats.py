# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:36:14 2018

@author: User
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial


def parallelize_dataframe(df, function, dictionnary, njobs):
    df_split = np.array_split(df, njobs)
    pool = Pool(njobs)
    func = partial(function, dictionnary)
    df2 = pd.concat(pool.map(func, df_split))
    
    pool.close()
    pool.join()
    
    return df2


def loop_size_data(x, sub_data):
     len_sub = 0
     i =2
     
     while len_sub ==0:
         index2 = (abs(sub_data["Date"] - x["Date"]).dt.days< 50*i)
         sub_data = sub_data.loc[index2]
            
         if len(sub_data.loc[~pd.isnull(sub_data["w_ace"])]) > 0:
             return sub_data
         else:
             if i <10:
                 i +=1
             else:
                 return []
    

def fillin_missing_stats(data_stats):
    
    data = data_stats.copy()
    
    ### fill in nans and 0 values for matches not retired or cancelled
    missing_data = data.loc[(pd.isnull(data["w_ace"]))|(pd.isnull(data["minutes"]))|(((data["l_svpt"] ==0)|(data["w_svpt"] ==0))&(data["status"] == "Completed"))].copy()
    
    for ind, i in enumerate(missing_data["ATP_ID"]):
        x = missing_data.iloc[ind]
        index1 = (data["winner_id"] == x["winner_id"])&(data["surface"] == x["surface"])&(data["best_of"] == x["best_of"])
        sub_data = data.loc[index1]
        sub_data1 = loop_size_data(x, sub_data)
        
        index2 = (data["loser_id"] == x["loser_id"])&(data["surface"] == x["surface"])&(data["best_of"] == x["best_of"])
        sub_data2 = data.loc[index2]
        sub_data2 =  loop_size_data(x, sub_data2) 
        
        #### no history around this match, we take average of stats for winners
        if pd.isnull(x["w_ace"]):
            if len(sub_data1) ==0:
             sub_data1 = data.loc[(abs(data["Date"] - x["Date"]).dt.days< 50)]
             
            for col in ["w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced"]:
                 data.loc[data["ATP_ID"] == i, col] = sub_data1[col].mean()
            
            if len(sub_data2) ==0:
                 sub_data2 = data.loc[(abs(data["Date"] - x["Date"]).dt.days< 50)]
            
            for col in ["l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced"]:
                data.loc[data["ATP_ID"] == i, col] = sub_data2[col].mean()

        if pd.isnull(x["minutes"]):
            data.loc[data["ATP_ID"] == i, "minutes"] = data.loc[((data["winner_id"] == x["winner_id"])|(data["loser_id"] == x["loser_id"]))&(data["best_of"] == x["best_of"]), "minutes"].mean()
   
    return data

