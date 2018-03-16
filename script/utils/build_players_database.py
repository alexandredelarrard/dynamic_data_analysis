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


if __name__ == "__main__":
    
    path = r"D:\projects\tennis betting\data\players\brute_info"
    liste_players = glob.glob(path + "/*.json")
    
    data_players = loop_over_jsons(liste_players)
    
    data_players["matching_name"] = data_players["Surname"] + data_players["Name"].apply(lambda x : " " + x[0:1] + ".")
#    
#
#    sub_data_players = data_players.loc[~data_players["matching_name"].isin(liste_duplicated)][["Name", "Surname", "matching_name"]]
#
#    path_players_list = r"D:\projects\tennis betting\data\players\liste_players.csv"
#    liste_players_need = pd.read_csv(path_players_list)
#    liste_players_need["players"] =  liste_players_need["players"].apply(lambda x : x.lstrip().rstrip())
#    liste_players_need["ID"] = liste_players_need.index
#    
#    players_merge = pd.merge(liste_players_need, sub_data_players, how = "left", left_on = "players", right_on= "matching_name")
#    players_merge[["ID", "Name", "Surname", "players", "matching_name"]].fillna("").to_csv(r"D:\projects\tennis betting\data\players\match_players_name.csv")
#    
#    missing_names = players_merge.loc[pd.isnull(players_merge["matching_name"]),"players"]