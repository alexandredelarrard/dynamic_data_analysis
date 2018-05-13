# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:36:14 2018

@author: User
"""

import pandas as pd
import numpy as np
import os

def loop_size_data(x, sub_data):
     len_sub = 0
     i =2
     
     while len_sub ==0:
         index2 = (abs(sub_data["Date"] - x["Date"]).dt.days< 50*i)
         sub_data = sub_data.loc[index2]
            
         if len(sub_data.loc[~pd.isnull(sub_data["w_ace"])]) > 0:
             return sub_data.loc[~pd.isnull(sub_data["w_ace"])]
         
         else:
             if i <10:
                 i +=1
                 len_sub = len(sub_data.loc[~pd.isnull(sub_data["w_ace"])])
             else:
                 return []
    

def fillin_missing_stats(data_stats):
    
    data = data_stats.copy()
    
    data.loc[pd.isnull(data["w_ace"])].to_csv(os.environ["DATA_PATH"] + "/to_check/missing_stats.csv")
    data["missing_stats"] = np.where(pd.isnull(data["w_ace"]), 1, 0)
    data.loc[data["l_svpt"] == 0, 'missing_stats'] = 1
    
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
        
        if pd.isnull(x["w_ace"]) or pd.isnull(x["l_ace"]):
            
            if len(sub_data1) ==0:
                sub_data1 = data.loc[(abs(data["Date"] - x["Date"]).dt.days< 50)]
             
            if pd.isnull(x["w_svpt"]) or x["w_svpt"] ==0:
                for col in ["w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced"]:
                    data.loc[data["ATP_ID"] == i, col] = sub_data1.loc[~pd.isnull(sub_data1[col]),col].astype(int).mean(skipna=True)
                    
            if len(sub_data2) ==0:
                 sub_data2 = data.loc[(abs(data["Date"] - x["Date"]).dt.days< 50)]
            
            if pd.isnull(x["l_svpt"]) or x["l_svpt"] ==0:
                for col in ["l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced"]:
                    data.loc[data["ATP_ID"] == i, col] =  sub_data2.loc[~pd.isnull(sub_data2[col]), col].astype(int).mean(skipna=True)

        if pd.isnull(x["minutes"]):
            data.loc[data["ATP_ID"] == i, "minutes"] = data.loc[((data["winner_id"] == x["winner_id"])|(data["loser_id"] == x["loser_id"]))&(data["best_of"] == x["best_of"]), "minutes"].mean()
       
    return data

