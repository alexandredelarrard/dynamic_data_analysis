# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 14:20:09 2018

@author: User
"""

import numpy as np
import json
import pandas as pd
import glob
import tqdm
import os

def loop_over_jsons(liste_jsons):
    first = 0
    data_players = pd.DataFrame([], columns = ["Name", "Surname", "Activity", "Rank", "Country", "age",  "DOB", "Turned pro", "Weight", "Height", \
                               "Birth place", "Residence place", "Plays", "Coach", "Stats"], index = range(len(liste_jsons)))
    for jsonpath in tqdm.tqdm(liste_jsons):
        json_extract = read_json(jsonpath)
        extracted = extract_json_into_frame(json_extract)

        if len(extracted) !=0:
            try:
                data_players.iloc[first] = extracted
                
            except Exception as e:
                    print(e)
                    print(extracted)
        else:
            print("no info for player %s"%jsonpath)
            
        first +=1
            
    return data_players


def extract_json_into_frame(json_extract):
    
    liste_keys = json_extract.keys()
    
    if 'global_desc' not in liste_keys:
        return [np.nan]*15
    
    else:
        desc= json_extract['global_desc']
        desc_global = [x for x in desc if x not in ['Ranking', 'Visit Official Site']]
        
        if not desc_global[3].isdigit():
            desc_global = desc_global[:3] + [np.nan] + desc_global[3:]
        
        if len(desc_global) == 13:
            desc_global = desc_global[:2] + ["#Singles#"] + desc_global[2:]
                                     
    if "specific_stats" in liste_keys:
        desc_stats = [json_extract['specific_stats']]
    else:
        desc_stats = [{"0":{}}]
        
    desc_all =  desc_global + desc_stats
    return np.array(desc_all)


def clean_data(data_players):
    data_players.drop(["Activity", "Rank", "age", "Coach"],axis=1, inplace= True)
    data_players["DOB"] = data_players["DOB"].apply(lambda x :  pd.to_datetime(x.replace("(","").replace(")",""), format = "%Y.%m.%d"))
   
    
def read_json(json_path):
    data = json.load(open(json_path))
    return data

def birthplace(x):
    
    if "," in x:
        return x.split(",")[-1]
    else:
        return x
    

def Plays_strong_hand(x):
    
    if "," in x:
        return x.split(",")[0].replace("Plays","").replace("Right Handed","Right-Handed").replace("Left Handed","Left-Handed")
    else:
        return x.replace("Plays", "")
    
def Plays_weak_hand(x):
    
    if "," in x:
        return x.split(",")[1]
    else:
        return np.nan


def clean_players_crawl(data_players):
    
    data_players["Country"] = data_players["Country"].apply(lambda x : str(x).rstrip().lstrip())
    data_players["Nationality"] = data_players["Country"].replace("nan", np.nan)
    del data_players["Country"]
    
    data_players["Birth place"] = data_players["Birth place"].apply(lambda x : x.replace("Birthplace","").lstrip())
    data_players["Birth place"] = data_players["Birth place"].apply(lambda x : birthplace(x))
    data_players.loc[(data_players["Birth place"] == ""), "Birth place"] = np.nan
    
    data_players["Weight"] = data_players["Weight"].apply(lambda x : x.replace("Weight","").lstrip())
    data_players["Weight"] = data_players["Weight"].apply(lambda x : x[x.find("(")+1:x.find(")")].replace("kg",""))
    data_players.loc[(data_players["Weight"] == "0") | (data_players["Weight"] == ""), "Weight"] = np.nan
    
    data_players["Height"] = data_players["Height"].apply(lambda x : x.replace("Height","").lstrip())
    data_players["Height"] = data_players["Height"].apply(lambda x : x[x.find("(")+1:x.find(")")].replace("cm",""))
    data_players.loc[(data_players["Height"] == "0") | (data_players["Height"] == ""), "Height"] = np.nan
    
    data_players["Turned pro"] = data_players["Turned pro"].apply(lambda x : x.replace("Turned Pro","").lstrip())
    data_players.loc[(data_players["Turned pro"] == "0") | (data_players["Turned pro"] == ""), "Turned pro"] = np.nan
    
    data_players["DOB"] = data_players["DOB"].apply(lambda x : x.replace(".","/").replace("(","").replace(")",""))
    data_players["DOB"] = data_players["DOB"].replace("DOB",np.nan)
    data_players["DOB"] = pd.to_datetime(data_players["DOB"], format = "%Y/%m/%d")
    
    data_players["Strong_hand"] = data_players["Plays"].apply(lambda x : Plays_strong_hand(x))
    data_players["Strong_hand"] = data_players["Strong_hand"].replace("",np.nan)
    
    data_players["Weak_hand"] = data_players["Plays"].apply(lambda x : Plays_weak_hand(x))
    
    del data_players["Plays"]
    
    data_players.to_csv(os.environ["DATA_PATH"] + "/brute_info/players/brute_info_players_dec.csv")
    
    return data_players

if __name__ == "__main__":
    
    liste_players = glob.glob(os.environ["DATA_PATH"] + "/brute_info/players/brute_info/players_descV2/*.json")
    data_players = loop_over_jsons(liste_players)
    
    data_players = data_players[["Name", "Surname", "Country", "DOB", "Turned pro", "Weight", "Height", "Birth place", "Plays"]]
    data_players["key"] = data_players[["Name", "Surname"]].apply(lambda x : x[0].lower().replace("-"," ").replace(".","").replace("'","").rstrip() + " " + x[1].lower().replace("-"," ").replace("."," ").replace("'","").rstrip(), axis=1)

    data_players = clean_players_crawl(data_players)
    
    players_db = pd.read_csv(os.environ["DATA_PATH"] + "/brute_info/players/players_ID_part2.csv")
    players_db["Player_Name"] = players_db["Player_Name"].str.lower()
    dd = pd.merge(players_db, data_players, left_on = "Player_Name", right_on ="key", how = "left")
    
    print(dd.loc[pd.isnull(dd["Name"])].shape)
    
    del dd["Unnamed: 0"]
    dd.to_csv(os.environ["DATA_PATH"] + "/clean_datasets/players/players_description_V2.csv", index= False)
