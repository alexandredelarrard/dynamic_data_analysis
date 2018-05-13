# -*- coding: utf-8 -*-
"""
Created on Sun May 13 09:39:37 2018

@author: User
"""

import pandas as pd
import os
import numpy as np


def merge_tourney(data):
    
    country = {'Australia' : "AUS", 'United States': "USA", 'New Zealand': "NZL", 'Brazil': "BRA", 'Italy':"ITA",
               'Belgium' : "BEL", 'Canada':"CAN", 'Germany':"GER", 'Netherlands': "NED", 'Morocco':"MAR",
               'Portugal' : "POR", 'Spain':"ESP", 'Japan':"JAP", 'France':"FRA", 'South Korea':"KOR", 'Hong Kong':"HKG",
               'Monaco':"MON", 'Singapore':"SGP", 'Croatia':"CRO", 'Great Britain':"GBR", 'Switzerland':"SUI",
               'Sweden':"SWE", 'Austria':"AUT", 'Czech Republic':"CZE", 'San Marino':"SAM", 'Greece':"GRE",
               'Israel':"ISR", 'Russia':"RUS", 'Denmark':"DEN", 'South Africa':"RSA",
               'Taiwan':"TPE", 'Qatar':"QAT", 'Malaysia':"MAS", 'Indonesia':"INA", 'United Arab Emirates':"UAE",
               'Mexico':"MEX", 'Romania':"ROM", 'China':"CHN", 'Chile':"CHI", 'Argentina':"ARG", 'Costa Rica':"CRC",
               'Colombia':"COL", 'Urugay':"URU", 'Bermuda':"BER", 'India':"IND", 'Uzbekistan':"UZB", 
               'Poland':"POL", 'Thailand':"THA", 'Vietnam':"VIE", 'Serbia':"SER", 'Ecuador':"ECU", 'Turkey':"TUR",
               'Bulgaria':"BUL", 'Hungary':"HUN"
                }
    
    ### create currency
    tournament = pd.read_csv(os.environ["DATA_PATH"] + "/clean_datasets/tournament/tourney.csv", encoding = "latin1")
    tournament.loc[pd.isnull(tournament["masters"]), "masters"] = "classic"
    tournament["Currency"] = np.where(tournament["Currency"] == "A", "AU$", 
                             np.where(tournament["Currency"] == 'Â£', "£", 
                             np.where(tournament["Currency"] == 'euro', "euro", "$")))
    
    ### homogenize and clean prize
    tournament["prize"] = tournament["prize"].apply(lambda x : x.replace("$","").replace(".",""))
    currency = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/tournament/currency_evolution.csv")
    tournament["prize"] = tournament[["prize", "Currency", "tourney_year"]].apply(lambda x: homogenize_prizes(x, currency), axis=1)

    data_merge = pd.merge(data, tournament, on = "tourney_id", how = "left")
    data_merge = data_merge.drop(["tourney_name", "surface_y", "tourney_id_atp", "tourney_year"], axis=1)
    
    data_merge.loc[data_merge["tourney_country"] == "Netherland", "tourney_country"] = "Netherlands"
    data_merge.loc[data_merge["tourney_country"] == "England", "tourney_country"] = "Great Britain"
    data_merge["tourney_country"] = data_merge["tourney_country"].map(country)
    
    return data_merge


def homogenize_prizes(x, currency):
    x["prize"] =  int(str(x["prize"]).replace("$","").replace(",",""))
    if x["Currency"] in ["AU$", "£", "$"]:
        x["prize"] = x["prize"]*currency.loc[currency["Annee"]==x["tourney_year"], x["Currency"]].values[0] / 1000000
    return  x["prize"]
    