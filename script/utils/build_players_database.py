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


if __name__ == "__main__":
    
    liste_players = glob.glob(os.environ["DATA_PATH"] + "/brute_info/players/brute_info/*.json")
    data_players = loop_over_jsons(liste_players)
    data_players = data_players[["Name", "Surname", "Country", "DOB", "Turned pro", "Weight", "Height", "Birth place", "Residence place", "Plays"]]
    
    data_players["key"] = data_players[["Name", "Surname"]].apply(lambda x : , axis=1)