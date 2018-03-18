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
