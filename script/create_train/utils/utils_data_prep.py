# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 09:00:49 2018

@author: User
"""
import pandas as pd
import numpy as np
from dateutil import relativedelta
from datetime import timedelta
import re

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

def correct_score(x):
    a = x.split()
    sortie = []
    for el in a:
        if el[0].isdigit() == True and el[1].isdigit() == True:
            sortie.append(str(el[0]) + "-" + str(el[1:]))
    return " ".join(sortie)

def status(x):
    if "RET" in x:
        return "Retired"
    elif "W/O" in x:
        return "Walkover"
    elif "DEF" in x:
        return "Def"
    else:
        return "Completed"
    
def liste1_extract(x):
    
    list2= x.split()
    end_date  = list2[-1]
    start_date = list2[-3]
    country = list2[-4]
    name =  " ".join(x.split(",")[0].split(" ")[:-1])
    city = x.split(",")[0].split(" ")[-1]
    return (end_date, start_date, country, name, city)
    

def currency_prize(x):
    for currency in ["€", "A$", "$", "£"]:
        s = x.split(currency)
        if len(s)>1:
            return currency, int(s[1].replace(",",""))
    return "", x.replace(",","")

def homogenize_prizes(x, currency):
    x["prize"] =  int(str(x["prize"]).replace("$","").replace(",",""))
    if x["Currency"] in ["AU$", "£", "$"]:
        x["prize"] = x["prize"]*currency.loc[currency["Annee"]==x["tourney_year"], x["Currency"]].values[0]
    return  x["prize"]

def calculate_time(x):
    x = x.replace(".00","")
    try:
         x = pd.to_datetime(str(x).replace("Time: ",""), format = "%H:%M:%S")
         time = x.hour * 60 + x.minute
         
    except Exception:
       return np.nan
    
    return time

def feature_round(x):
    
    if x[0] in ["Finals", "F"]:
        return 2/x[1]
    
    if x[0] in ["Semi-Finals", "SF"]:
        return 4/x[1]
    
    if x[0] in ["Quarter-Finals", "QF"]:
        return 8/x[1]
    
    if x[0] in ["Round Robin", "RR", "BR"]:
        return x[1]/x[1]
    
    if x[0] in ["Round of 16", "R16"]:
        return 16/x[1]
    
    if x[0] in ["Round of 32", "R32"]:
        return 32/x[1]
    
    if x[0] in ["Round of 64", "R64"]:
        return 64/x[1]
    
    if x[0] in ["Round of 128", "R128", "R96"]:
        return 128/x[1]
    
    if x[0] in ["Q-1st", "Q-2nd"]:
        return 1.5
    
    return np.nan
    
 
def dates(x):
    diff = relativedelta.relativedelta(x[0], x[1])
    years = diff.years
    reste = (diff.months *30 + diff.days) / 365
    return years + reste

def extract_rank_and_match(x, rk_data):
    """
    match rank with player name loser and winner based on closest date into the past
    """
    
    dates = pd.to_datetime(rk_data.sort_values("Date")["Date"].unique())
    date = dates[dates <= x["Date"]][-1]
    rank_sub_df = rk_data.loc[rk_data["Date"] == date]
    
    try:
        winner = rank_sub_df.loc[rank_sub_df["Player_name"] == x["winner_name"]][["player_rank", "player_points"]].values[0]
    except Exception:
        print(x)
        print(rank_sub_df.loc[rank_sub_df["Player_name"] == x["loser_name"]][["player_rank", "player_points"]])
        winner = [1800, 0]    
        
    try:
        loser =  rank_sub_df.loc[rank_sub_df["Player_name"] == x["loser_name"]][["player_rank", "player_points"]].values[0]
    except Exception:
        print(x)
        print(rank_sub_df.loc[rank_sub_df["Player_name"] == x["loser_name"]][["player_rank", "player_points"]])
        loser = [1800, 0]
        
    return [(winner[0],winner[1],loser[0],loser[1])]