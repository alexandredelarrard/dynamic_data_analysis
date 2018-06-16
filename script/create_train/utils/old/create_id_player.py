# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 15:55:02 2018

@author: JARD
"""

import pandas as pd
import numpy as np
import os


def players_ID_creation(data_merge):
    
   players_db = pd.DataFrame(np.transpose([list(data_merge["winner_id"]) + list(data_merge["loser_id"]), list(data_merge["winner_name"])+ list(data_merge["loser_name"])]), columns = ["Players_ID", "Player_Name"])
   players_db = players_db.drop_duplicates()

   players_db.to_csv(os.environ["DATA_PATH"] + "/brute_info/players/players_ID.csv")
   
   print("Dataset of players ID created for {0}".format(players_db.shape[0]))



def players_ID_creation_additional(data_merge):
    
   players_db = pd.DataFrame(np.transpose([list(data_merge["winner_id"]) + list(data_merge["loser_id"]), list(data_merge["winner_name"])+ list(data_merge["loser_name"])]), columns = ["Players_ID", "Player_Name"])
   players_db = players_db.drop_duplicates()
   players_db["Players_ID"] = players_db["Players_ID"].astype(int)
   
   already_there = pd.read_csv(r"C:\Users\User\Documents\tennis\data\clean_datasets\players/players_desc.csv")
   
   players_id_to_add = set(players_db["Players_ID"]) - set(already_there["Players_ID"])
   players_db = players_db.loc[players_db["Players_ID"].isin(list(players_id_to_add))]
   
   
#   data_merge[["winner_id", "winner_ioc"]
   
   players_db.to_csv(os.environ["DATA_PATH"] + "/brute_info/players/players_ID_part2.csv")
   
   
   
   players_db["Nationality"] = 