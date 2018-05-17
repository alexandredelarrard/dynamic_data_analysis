# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:47:25 2018

@author: User
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os

def calculate_corr_time(tot, start_year=1990, end_year=2016, weight_1 = 3, redo = False):
    
    if redo:
        def exponenial_func(x, a, b, c):
            return a*np.exp(-b*x)+c
    
        tot_loc = tot.loc[(tot["Date"].dt.year >= start_year)&(tot["Date"].dt.year <= end_year)].copy()
        tot_loc["year"] = tot_loc["Date"].dt.year
        tot_loc["month"] = tot_loc["Date"].dt.month
        
        liste_players = list(set(tot_loc["winner_id"].tolist() + tot_loc["winner_id"].tolist()))
        dataframe = pd.DataFrame([], index= liste_players, columns = range(22))
        
        for pl in liste_players: 
            pl_data = tot_loc.loc[tot_loc["winner_id"] == pl].copy()
            
            agg = pl_data[["year", "month", "target"]].groupby(["year", "month"]).mean()
            if len(agg) > 5:
                rep = agg["target"].tolist()[::-1]
                for i in range(len(agg)):
                    dataframe.loc[pl, i] = rep[i]
                    
        dataframe[dataframe.columns] = dataframe[dataframe.columns].astype(float)
        time_correlations = dataframe.corr()
        
        #### first 3 months should have weight = 1 ---> leads to first 12 months weight = 1
        count_1 = weight_1
        size = 150
        time_correlations = time_correlations.loc[:size, :size]
        
        time_corr = []
        for t in range(1,size):
            s = 0
            for i in range(size):
                if time_correlations.shape[1] > i+t:
                    s  += time_correlations.iloc[i, i+t]
                    
            time_corr.append(s / size)
            
        coef_offset = 1/time_corr[count_1]    
        corr_temp = pd.DataFrame(time_corr)*coef_offset
        corr_temp.index = list(range(1, len(corr_temp)+1))
        popt, pcov = curve_fit(exponenial_func, corr_temp.index, corr_temp[0], p0=(1, 1e-2, 1))
        
        a = []
        for x in range(1,size):
            a.append(exponenial_func(x, popt[0], popt[1], popt[2]))
        
        corr_temp["pred"] = a
        corr_temp.loc[corr_temp["pred"]>=1, "pred"] = 1
        corr_temp.loc[corr_temp["pred"]<=0, "pred"] = 0
        corr_temp[[0,"pred"]].plot()
        
        corr_temp[0] = 1
        
        corr_temp.to_csv(os.environ["DATA_PATH"] + "/clean_datasets/historical/correlations_temps.csv")
    
    else:
        corr_temp = pd.read_csv(os.environ["DATA_PATH"] + "/clean_datasets/historical/correlations_temps.csv")
    
    return corr_temp["pred"]


def calculate_corr_surface(tot, start_year=1990, end_year=2016, redo = False):
    
    ### because dont want info from test set
    if redo:
        tot_loc = tot.loc[(tot["Date"].dt.year >= start_year)&(tot["Date"].dt.year <= end_year)]
        liste_players = list(set(tot_loc["winner_id"].tolist() + tot_loc["winner_id"].tolist()))
        dataframe = pd.DataFrame([], index= liste_players, columns = tot["surface"].unique())
        
        for pl in liste_players: 
            pl_data = tot_loc.loc[tot_loc["winner_id"] == pl]
            agg = pl_data[["surface", "target"]].groupby("surface").mean()
            for i, surf in enumerate(agg.index.tolist()):
                dataframe.loc[pl, surf] = agg.loc[surf].values[0]
        
        dataframe = dataframe.loc[(~pd.isnull(dataframe[tot["surface"].value_counts().index[0]]))]
        
        dataframe[dataframe.columns] = dataframe[dataframe.columns].astype(float)
        surface_correlations = dataframe.corr()
        
        surface_correlations.to_csv(os.environ["DATA_PATH"] + "/clean_datasets/historical/correlations_surface.csv")
        
    else:
        surface_correlations = pd.read_csv(os.environ["DATA_PATH"] + "/clean_datasets/historical/correlations_surface.csv", index_col=0)
    
    return surface_correlations


def proportion_win(x, total_data):
    
    sub_data1 = total_data.loc[(total_data["Date"] < x["Date"])&(total_data["winner_id"] == x["winner_id"])].copy()
    sub2 = sub_data1.loc[(sub_data1["loser_id"] == x["loser_id"])].copy()
    
    liste_id_player1 = set(sub_data1["loser_id"])
    liste_id_player2 = set(total_data.loc[(total_data["Date"] < x["Date"])&(total_data["winner_id"] == x["loser_id"]), "loser_id"].copy())
    
    sub3 = sub_data1.loc[sub_data1["loser_id"].isin(list(set.intersection(liste_id_player1, liste_id_player2)))].copy()
    
    if len(sub3)>0 and len(sub2)>0:
        return [[sub_data1["target"].mean(), sub2["target"].mean(), sub3["target"].mean()]]   
    else:
        return [[sub_data1["target"].mean(), np.nan, np.nan]]    


def calculate_corr_opponents(tot, remake = False, start_year=1990, end_year=2016):
    
    if remake:
        tot_loc = tot.loc[(tot["Date"].dt.year >= start_year)&(tot["Date"].dt.year <= end_year)].copy()
        data = tot_loc[["Date","winner_id", "loser_id", "target"]]
        
        rep = data.apply(lambda x : proportion_win(x, data), axis=1)["loser_id"]
        a = list(list(zip(*rep))[0])
        b = list(list(zip(*rep))[1])
        c = list(list(zip(*rep))[2])
        
        d = pd.DataFrame(np.transpose([a,b,c]), columns= ["1Vall", "1v1", "1Vsub"])
        corr = d.loc[~pd.isnull(d["1Vall"])]
        
        return corr
        
    else:
        return pd.DataFrame([1, 0.9, 0.3], index= ["1v1", "1Vsub", "1Vall"], columns = ["correlation"])
    
    