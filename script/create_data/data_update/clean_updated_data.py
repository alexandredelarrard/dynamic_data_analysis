# -*- coding: utf-8 -*-
"""
Created on Mon May 28 13:40:50 2018

@author: JARD
"""

import pandas as pd
import numpy as np
import sys
import os
from dateutil import relativedelta
from datetime import timedelta
import glob

sys.path.append(r"C:\Users\User\Documents\tennis\dynamic_data_analysis\script")
from create_data.data_creation.create_variables import prep_data
from create_data.data_creation.extract_players import import_players
from create_data.utils.date_creation import deduce_match_date


def currency_prize(x):
    
    for currency in ["€", "A$", "$", "£"]:
        s = x.split(currency)
        if len(s)>1:
            return currency, int(s[1].replace(",",""))
    return "", x.replace(",","")


def liste1_extract(x):
    
    liste= eval(x)
    list2= liste[0].split()
    draw_size = liste[1].split()[1]
    end_date  = list2[-1]
    start_date = list2[-3]
    country = list2[-4]
    city = liste[0].split(",")[0].split(" ")[-1]
    name = " ".join(liste[0].split(",")[0].split(" ")[:-1])
    master = "" if liste[-1] == "atpwt" else liste[-1]
    
    return (draw_size, end_date, start_date, country, city, name, master)


def calculate_time(x):
    x = x.replace(".00","")
    try:
         x = pd.to_datetime(str(x).replace("Time: ",""), format = "%H:%M:%S")
         time = x.hour * 60 + x.minute
         
    except Exception:
       return np.nan
    
    return time


def dates(x):
    diff = relativedelta.relativedelta(x[0], x[1])
    years = diff.years
    reste = (diff.months *30 + diff.days) / 365
    return years + reste


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


def homogenize_prizes(x, currency):
    x["prize"] =  int(str(x["prize"]).replace("$","").replace(",",""))
    if x["Currency"] in ["AU$", "£", "$"]:
        x["prize"] = x["prize"]*currency.loc[currency["Annee"]==x["tourney_year"], x["Currency"]].values[0]
    return  x["prize"]


def clean_extract(latest):
    """
    clean the new data extracted from atp website
    """
    
    dico_round = {"Round of 32": "R32", "Round of 16": "R16", "Round of 64": "R64", "Round of 128":"R128", "Finals": "F", "Semi-Finals":"SF", "Quarter-Finals":"QF", "Round Robin":"RR", "Round of 96":"R96"}
    extract = pd.read_csv(os.environ["DATA_PATH"] + "/clean_datasets/overall/updated/extracted/extraction_brute.csv")
    
    ex = extract.copy()
    ex.columns = [str(x) for x in ex.columns]
    clean = pd.DataFrame([])
    
    # =============================================================================
    #     ### extraction of clean info
    # =============================================================================
    clean["indoor_flag"] = ex["1"].apply(lambda x : x.split("\r")[0])
    
    clean["surface"] = ex["1"].apply(lambda x :x.split("\n")[1])
    clean["winner_name"] = ex["8"].apply(lambda x : x.lower().replace("-"," ").lstrip().rstrip())
    clean["loser_name"] = ex["9"].apply(lambda x : x.lower().replace("-"," ").lstrip().rstrip())
    clean["status"] = ex["10"].apply(lambda x: status(x))
    
    clean["score"]     =  ex["10"].apply(lambda x: correct_score(x))
    clean["match_num"] =  ex["11"]
    
    list_info1 = ex["7"].apply(lambda x : liste1_extract(x))
    clean["draw_size"] = list(list(zip(*list_info1))[0])
    clean["draw_size"] = clean["draw_size"].astype(int)
    clean["tourney_end_date"] = pd.to_datetime(list(list(zip(*list_info1))[1]), format="%Y.%m.%d") 
    clean["tourney_date"] = pd.to_datetime(list(list(zip(*list_info1))[2]), format="%Y.%m.%d")
    clean["tourney_country"] = list(list(zip(*list_info1))[3])
    clean["tourney_city"] = list(list(zip(*list_info1))[4])
    clean["tourney_name"] = list(list(zip(*list_info1))[5])
    clean["masters"] = list(list(zip(*list_info1))[6])
    clean["round"] = ex["64"].str.replace(" H2H", "")
    clean["round"] = clean["round"].map(dico_round)
    clean["tourney_id"] = clean["tourney_date"].dt.year.astype(str) + "-" + ex["6"].astype(str) 
    clean["Date"] = clean[["tourney_date", "tourney_end_date", "match_num", "draw_size", "round"]].apply(lambda x: deduce_match_date(x), axis=1)
    clean["tourney_year"] = clean["Date"].dt.year
    clean["best_of"] = np.where(clean["masters"] == "grandslam", 5, 3)
    
    count = ex["2"].apply(lambda x : currency_prize(x))
    
    # =============================================================================
    #     ### take care currency 
    # =============================================================================
    clean["Currency"] = list(list(zip(*count))[0])
    clean["Currency"] = np.where(clean["Currency"].isin(["euros","€"]), "euro", clean["Currency"])
    clean["Currency"] = np.where(clean["Currency"].isin(["AUS$","AU$","A$"]), "AU$", clean["Currency"])
    clean["Currency"] = np.where(clean["Currency"].isin(["A£","'Â£'","£"]), "£", clean["Currency"])
    
    # =============================================================================
    #     ### homogenize price
    # =============================================================================
    clean["prize"] = list(list(zip(*count))[1])
    currency = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/tournament/currency_evolution.csv")
    clean["prize"] = clean[["prize", "Currency", "tourney_year"]].apply(lambda x: homogenize_prizes(x, currency), axis=1)
    
    # =============================================================================
    #     #### merge player id 
    # =============================================================================
    players = import_players()
    
    clean = pd.merge(clean, players, left_on = "winner_name", right_on = "Player_Name", how = "left")
    clean = pd.merge(clean, players, left_on = "loser_name", right_on = "Player_Name", how = "left", suffixes= ("_w","_l"))
    clean = clean.drop(["Player_Name_w", "Player_Name_l"], axis=1)
    
    clean = clean.rename(columns = {"Height_l": "loser_ht", 
                                    "Height_w": "winner_ht", 
                                    'Strong_hand_w': "winner_hand", 
                                    'Strong_hand_l': "loser_hand",
                                    'Nationality_w':"winner_ioc",
                                    'Nationality_l':"loser_ioc"})
    
    clean["winner_age"] = clean[["Date", "DOB_w"]].apply(lambda x : dates(x), axis=1)
    clean["loser_age"]  = clean[["Date", "DOB_l"]].apply(lambda x : dates(x), axis=1)
    
    #### stats
    ex_stats = ex.iloc[:,12:]
    ex_stats.columns= [str(x) for x in range(ex_stats.shape[1])]
    ex_stats = ex_stats.rename(columns = {"0":"url", "2": "sr_player1", "4": "sr_player2", 
                        "5": "aces_player1", "7" :"aces_player2", "8":"df_player1", "10":"df_player2", 
                        "11": "1serv_player1", "13": "1serv_player2", "14": "1serv_won_player1", "16": "1serv_won_player2",
                        "17": "2serv_won_player1", "19": "2serv_won_player2", "20" : "bp_saved_player1", "22" : "bp_saved_player2",
                        "23": "serv_games_player1", "25": "serv_games_player2", "27" : "rr_player1", "29" : "rr_player2",
                        "30": "1serv_return_won_player1", "32": "1serv_return_wplayerson_player2", "33": "2serv_return_won_player1", "35": "2serv_return_won_player2",
                        "36": "bp_converted_player1", "38": "bp_converted_player2", "39": "return_games_player1", "41": "return_games_player2",
                        "53": "player1", "54" : "score_player1", "55":"player2", "56": "score_player2", "57": "time"})
    
    clean["minutes"] = ex_stats["time"].fillna("0:0:0").apply(lambda x :calculate_time(x))
    clean["minutes"] = clean["minutes"].replace(0, np.nan).astype(int)
    
    clean["w_ace"] = ex_stats["aces_player1"].astype(int)
    clean["l_ace"] = ex_stats["aces_player2"].astype(int)
    clean["w_df"] = ex_stats["df_player1"].astype(int)
    clean["l_df"] = ex_stats["df_player2"].astype(int)
    
    clean["w_svpt"] = ex_stats["1serv_player1"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[1])
    clean["l_svpt"] = ex_stats["1serv_player2"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[1])
    clean["w_1stIn"] = ex_stats["1serv_player1"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    clean["l_1stIn"] = ex_stats["1serv_player2"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    clean["w_1stWon"] = ex_stats["1serv_won_player1"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    clean["l_1stWon"] = ex_stats["1serv_won_player2"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    clean["w_2ndWon"] = ex_stats["2serv_won_player1"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    clean["l_2ndWon"] = ex_stats["2serv_won_player2"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    clean["w_SvGms"] = ex_stats["serv_games_player1"].astype(int)
    clean["l_SvGms"] = ex_stats["serv_games_player2"].astype(int)
    
    clean["w_bpSaved"] = ex_stats["bp_saved_player1"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    clean["l_bpSaved"] = ex_stats["bp_saved_player2"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[0])
    clean["w_bpFaced"] = ex_stats["bp_saved_player1"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[1])
    clean["l_bpFaced"] = ex_stats["bp_saved_player2"].apply(lambda x : x.split("(")[1].split(")")[0].split("/")[1])
    
    # =============================================================================
    #     #### data prep/ craetion of variables
    # =============================================================================
    clean2 = prep_data(clean.loc[clean["status"] == "Completed"])
    
    # =============================================================================
    #     #### add rank data into it
    # =============================================================================
    files_rank = glob.glob(os.environ["DATA_PATH"] + "/brute_info/atp_ranking/*.csv")
    files_df = pd.DataFrame(np.transpose([files_rank, [pd.to_datetime(os.path.splitext(os.path.basename(x))[0], format = "%Y-%m-%d") for x in files_rank]]), columns = ["file", "Date"])
    files_rank = files_df.loc[files_df["Date"] >= pd.to_datetime(latest["Date"], format = "%Y-%m-%d") - timedelta(days=30)]
    
    for i, f in enumerate(files_rank["file"].tolist()):
        if i ==0:
            rk_data = pd.read_csv(f)
            rk_data["Date"] = files_rank.iloc[i,1]
        else:
            new_data = pd.read_csv(f)
            new_data["Date"] = files_rank.iloc[i,1]
            rk_data = pd.concat([rk_data, new_data],axis =0)
    rk_data= rk_data.reset_index(drop=True)
    rk_data["player_rank"] = rk_data["player_rank"].str.replace("T","").astype(int)
    rk_data["Player_name"] = rk_data["Player_name"].apply(lambda x : x.replace("-"," ").lower())
    
    count = clean2[["Date", "winner_name", "loser_name"]].apply(lambda x: extract_rank_and_match(x, rk_data), axis =1)["Date"]
    
    for i, col in enumerate(["winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points"]):
        clean2[col]  = list(list(zip(*count))[i])
        
    clean2 = clean2.rename(columns = {"Players_ID_w":"winner_id", "Players_ID_l":"loser_id"})
    
    return clean2


if __name__ == "__main__":
    latest = {"Date":"2018-02-05"}
    clean = clean_extract(latest)