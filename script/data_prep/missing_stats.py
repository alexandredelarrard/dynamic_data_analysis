# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:36:14 2018

@author: User
"""

import pandas as pd


def loop_size_data(x, sub_data):
     len_sub = 0
     i =2
     
     while len_sub ==0:
         index2 = (sub_data["best_of"] == x["best_of"])&(abs(sub_data["Date"] - x["Date"]).dt.days< 50*i)
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
    missing_data = data_stats.loc[pd.isnull(data_stats["w_ace"])].copy()
    
    for i in missing_data["ATP_ID"]:
        x = missing_data.loc[missing_data["ATP_ID"] ==i]
        x = x.iloc[0]
        index1 = (data["winner_id"] == x["winner_id"])&(data["surface"] == x["surface"])
        sub_data = data.loc[index1]
        sub_data1 = loop_size_data(x, sub_data)
        
        #### no history around this match, we take average of stats for winners
        if len(sub_data1) ==0:
             sub_data1 = data.loc[(abs(data["Date"] - x["Date"]).dt.days< 50)]
       
        if pd.isnull(x["w_ace"]):
            for col in ["w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced"]:
                 data_stats.loc[data_stats["ATP_ID"] == i, col] = sub_data1[col].mean()
    
        index1 = (data["loser_id"] == x["loser_id"])&(data["surface"] == x["surface"])
        sub_data = data.loc[index1]
        sub_data1 =  loop_size_data(x, sub_data)      
        
        if len(sub_data1) ==0:
             sub_data1 = data.loc[(abs(data["Date"] - x["Date"]).dt.days< 50)]
        
        if pd.isnull(x["l_ace"]):
            for col in ["l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced"]:
                data_stats.loc[data_stats["ATP_ID"] == i, col] = sub_data1[col].mean()

        if pd.isnull(x["minutes"]):
            data.loc[data_stats["ATP_ID"] == i, "minutes"] = data.loc[((data["winner_id"] == x["winner_id"])|(data["loser_id"] == x["loser_id"]))&(data["best_of"] == x["best_of"]), "minutes"].mean()
   
    return data_stats

