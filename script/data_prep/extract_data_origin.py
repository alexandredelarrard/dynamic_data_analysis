# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:11:36 2018

@author: JARD
"""

import pandas as pd
import glob
import numpy as np
from datetime import datetime
import os

def Number_rounds(data):

    data["Nbr_rounds"] = 6
    for year in range(2000,2019):
        sub_year = data.loc[data.Date.dt.year == year]
        for tournois in sub_year.Tournament.value_counts().index:
            data.loc[(data["Tournament"] == tournois) & (data.Date.dt.year == year), "Nbr_rounds"] = len(sub_year.loc[sub_year["Tournament"] == tournois, "Round"].value_counts())
    
    return data


def round_encoding(x):
    
    try:
        x[0] = x[0].replace("Quarterfinals", "0.25").replace("Semifinals", "0.5").replace("The Final","1").replace("Round Robin", "0")
        
        if x[1] == 7:
            x[0] = x[0].replace("1st Round", "0.015625").replace("2nd Round", "0.03125").replace("3rd Round", "0.0625").replace("4th Round", "0.125")
        
        if x[1] == 6:
            x[0] = x[0].replace("1st Round", "0.03125").replace("2nd Round", "0.0625").replace("3rd Round", "0.125")
            
        if x[1] == 5:
            x[0] = x[0].replace("1st Round", "0.0625").replace("2nd Round", "0.125").replace("3rd Round", "0.25")     
            
        if x[1] == 4:
            x[0] = x[0].replace("1st Round", "0.125").replace("2nd Round", "0.25").replace("3rd Round", "0.5")  
            
        if x[1] == 3:
            x[0] = x[0].replace("1st Round", "0.25").replace("2nd Round", "0.5")  .replace("3rd Round", "1")  
            
        if x[1] == 1:
            x[0] = x[0].replace("1st Round", "0")
            
    except Exception:
        print(x)
        
    return x[0]


def fill_missing_rank(data):
    
    ### correct winner ranks
    data.loc[(data["Date"] == "2003-07-28")&(data["Winner"] == "Bryan M."), "WRank"] = 655
    data.loc[(data["Date"] == "2003-07-30")&(data["Winner"] == "Bryan M."), "WRank"] = 655
    data.loc[(data["Date"] == "2004-01-05")&(data["Winner"] == "Johansson T."), "WRank"] = 519
    data.loc[(data["Date"] == "2004-01-07")&(data["Winner"] == "Johansson T."), "WRank"] = 519
    data.loc[(data["Date"] == "2006-01-31")&(data["Winner"] == "Friedl L."), "WRank"] = 615
    data.loc[(data["Date"] == "2006-02-02")&(data["Winner"] == "Friedl L."), "WRank"] = 615
    data.loc[(data["Date"] == "2007-06-19")&(data["Winner"] == "Allegro Y."), "WRank"] = 817
    data.loc[(data["Date"] == "2008-02-12")&(data["Winner"] == "Coria G."), "WRank"] = 809
    data.loc[(data["Date"] == "2010-01-05")&(data["Winner"] == "El Aynaoui Y."), "WRank"] = 735
    data.loc[(data["Date"] == "2012-01-31")&(data["Winner"] == "Mathieu P.H."), "WRank"] = 733
    data.loc[(data["Date"] == "2013-09-16")&(data["Winner"] == "Khachanov K."), "WRank"] = 805
    data.loc[(data["Date"] == "2015-04-07")&(data["Winner"] == "Tipsarevic J."), "WRank"] = 829
    data.loc[(data["Date"] == "2015-04-12")&(data["Winner"] == "Mayer F."), "WRank"] = 609
    data.loc[(data["Date"] == "2015-06-09")&(data["Winner"] == "Haas T."), "WRank"] = 850
    data.loc[(data["Date"] == "2016-07-12")&(data["Winner"] == "Wessels L."), "WRank"] = 620
    data.loc[(data["Date"] == "2000-03-23")&(data["Winner"] == "Fish M."), "WRank"] = 711

    ### correct loser
    missing_data= pd.read_csv(os.environ["DATA_PATH"] + "/brute_info/historical/correct_missing_values/missing_rank_loser.csv")
    for date, name in np.array(data.loc[pd.isnull(data["LRank"])][["Date", "Loser"]]):
        data.loc[(data["Date"] == date)&(data["Loser"] == name), "LRank"]  = missing_data.loc[(missing_data["Date"] == date) &(missing_data["Loser"] == name), "Rank"]
        
    data.loc[data["LRank"] == 'NR', "LRank"] = 1800 
    
    data["WRank"] = data["WRank"].astype(int)
    data["LRank"] = data["LRank"].astype(int)
    data["best_of"] = data["best_of"].astype(int)
        
    return data


def import_data_origin(path):
    
    liste_files = glob.glob(path + "/*.xls") + glob.glob(path + "/*.xlsx")
    
    for i, file in enumerate(liste_files):
        if i == 0:
            data = pd.read_excel(file)
        else:
            data = pd.concat([data, pd.read_excel(file)], axis=0)
            
    ### take care of error of tournament 2001 - 2004
    data.loc[(data["Tournament"] == "Heineken Open") & (data["Date"].dt.month != 1), "Tournament"] = "Shanghai Open"
    data.loc[data["Tournament"] == "Heineken Open Shanghai", "Tournament"] = "Shanghai Open"
    
    ### take care of error for final 2003 ca tennis trophy
    data.loc[(data["Tournament"] == "CA Tennis Trophy") & (data["Date"] == "2003-11-12"), "Date"] = datetime.strptime("2003-10-12", "%Y-%m-%d")
        
    ### take care of error for final 2017 sony ericson
    data.loc[(data["Tournament"] == "Sony Ericsson Open") & (data["Date"] == "2017-01-02"), "Date"] = datetime.strptime("2017-04-02", "%Y-%m-%d")
    
    #### take care of error on outdoor 
    data.loc[(data["Tournament"] == "Kingfisher Airlines Tennis Open") & (data["Date"].dt.year.isin([2006,2007])), "Court"] = "Outdoor"
    data.loc[(data["Tournament"] == "SA Tennis Open") & (data["Date"].dt.year ==2011), "Court"] = "Indoor"
    data.loc[(data["Tournament"] == "Brasil Open") & (data["Date"].dt.year.isin([2016,2017])), "Court"] = "Outdoor"
    
    ### encode rounds as float
    data = Number_rounds(data)
    data["Round"] = data[["Round", "Nbr_rounds"]].apply(lambda x : round_encoding(x), axis=1)
    data["Round"] = data["Round"].astype(float)
    
    ### replace missing ranks
    data = fill_missing_rank(data)
    
    ### TAKE CARE OF SETS
    data.loc[data["Lsets"] == "`1", "Lsets"] = "1"
    data.loc[pd.isnull(data["Lsets"]), "Lsets"] = -1
    data["Lsets"] = data["Lsets"].astype(int)
    
    data.loc[pd.isnull(data["Wsets"]), "Wsets"] = -1
    data["Wsets"] = data["Wsets"].astype(int)
    
    data["ORIGIN_ID"] = range(len(data))
    
    return data.reset_index(drop=True)


