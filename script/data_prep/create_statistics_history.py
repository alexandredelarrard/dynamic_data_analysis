# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:35:39 2018

@author: User
"""

### data come from http://www.tennis-data.co.uk/alldata.php
import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial
from datetime import  timedelta
import time


def parallelize_dataframe(df, function, dictionnary, njobs):
    df_split = np.array_split(df, njobs)
    pool = Pool(njobs)
    func = partial(function, dictionnary)
    df2 = pd.concat(pool.map(func, df_split))
    
    pool.close()
    pool.join()
    
    return df2

def create_stats(dico, data): 
    count =  data[["Winner", "Loser", "Surface", "Series", "ATP", "Date", 'Court']].apply(lambda x : basic_history_statistics(x, dico), axis= 1)
    return count["Winner"]


def basic_history_statistics(x, dico):
    """
    x : 0 = Winner 
        1 = Loser
        2 = Surface
        3 = Series
        4 = Tournois
        5 = Date
        6 = 'Court'
    dico : whole data
    """
    
    df_train= dico[0]
    res= ()
    
    ### 
    data_past_players = df_train.loc[((df_train['Winner'] == x[0]) & (df_train['Loser'] == x[1])) | ((df_train['Winner'] == x[1]) & (df_train['Loser'] == x[0]))].copy()
    data_past_player1 = df_train.loc[((df_train['Winner'] == x[0]) | (df_train['Loser'] == x[0]))].copy()
    data_past_player2 = df_train.loc[((df_train['Winner'] == x[1]) | (df_train['Loser'] == x[1]))].copy()
   
    
    ### between players
    for jours in [100, 400, 1500]:
        
        data_past_players = data_past_players.loc[(data_past_players["Date"] < x[5])&(data_past_players["Date"] >= (pd.to_datetime(x[5]) - timedelta(days=jours)))]
        data_past_player1 = data_past_player1.loc[(data_past_player1["Date"] < x[5])&(data_past_player1["Date"] >= (pd.to_datetime(x[5]) - timedelta(days=jours)))]
        data_past_player2 = data_past_player2.loc[(data_past_player2["Date"] < x[5])&(data_past_player2["Date"] >= (pd.to_datetime(x[5]) - timedelta(days=jours)))]
        
        res += basic_statistics(data_past_players, data_past_player1, data_past_player2, x)
    
        
    return [res]


def basic_statistics(data_past_players, data_past_player1, data_past_player2, x):

    stat = (
            data_past_player1.shape[0], ### nbr match joues player 1
            data_past_player2.shape[0], ### nbr match joues player 2
            
            sum(((data_past_player1["Winner"] == x[0]))*1), ### nbr match gagne player 1
            sum(((data_past_player2["Winner"] == x[1]))*1), ### nbr match gagne player 2
            
            data_past_player1.loc[data_past_player1["Surface"] == x[2]].shape[0], ### nbr match joues player 1 meme surface
            data_past_player2.loc[data_past_player2["Surface"] == x[2]].shape[0], ### nbr match joues player 2 meme surface
            
            sum(((data_past_player1["Winner"] == x[0])&(data_past_player1["Surface"] == x[2]))*1), ### nbr match gagne player 1 meme surface
            sum(((data_past_player2["Winner"] == x[1])&(data_past_player2["Surface"] == x[2]))*1), ### nbr match gagne player 2 meme surface
            
            data_past_player1.loc[data_past_player1["Surface"] == x[3]].shape[0], ### nbr match joues player 1 meme serie
            data_past_player2.loc[data_past_player2["Surface"] == x[3]].shape[0], ### nbr match joues player 2 meme serie
            
            sum(((data_past_player1["Winner"] == x[0])&(data_past_player1["Surface"] == x[3]))*1), ### nbr match gagne player 1 meme serie
            sum(((data_past_player2["Winner"] == x[1])&(data_past_player2["Surface"] == x[3]))*1), ### nbr match gagne player 2 meme serie
            
            min(list(data_past_player1.loc[(data_past_player1['Winner'] == x[0]) , "Wrank"]) + list(data_past_player1.loc[(data_past_player1['Loser'] == x[0]), "Lrank"])),  ### best rank joueur 1
            min(list(data_past_player2.loc[(data_past_player2['Winner'] == x[1]) , "Wrank"]) + list(data_past_player2.loc[(data_past_player2['Loser'] == x[1]), "Lrank"])),  ### best rank joueur 2
             
            max(list(data_past_player1.loc[(data_past_player1['Winner'] == x[0]) , "Wrank"]) + list(data_past_player1.loc[(data_past_player1['Loser'] == x[0]), "Lrank"])), ### worst rank joueur 1
            max(list(data_past_player2.loc[(data_past_player2['Winner'] == x[1]) , "Wrank"]) + list(data_past_player2.loc[(data_past_player2['Loser'] == x[1]), "Lrank"])), ### worst rank joueur 1
            
            sum(data_past_player1.loc[(data_past_player1['Loser'] == x[0]) , "Lsets"]),  ### nbr set gagne quand perdu player1
            sum(data_past_player2.loc[(data_past_player2['Loser'] == x[1]) , "Lsets"]),  ### nbr set gagne quand perdu player2
            
            sum(data_past_player1.loc[(data_past_player1['Winner'] == x[0]) , "Lsets"]),  ### nbr set concede quand gagne player 1
            sum(data_past_player2.loc[(data_past_player2['Winner'] == x[1]) , "Lsets"]),  ### nbr set concede quand gagne player 1
            
            data_past_player1.loc[data_past_player1['Court'] == x[6]].shape[0], ### nb match joue meme int/ext player1 
            data_past_player2.loc[data_past_player2['Court'] == x[6]].shape[0], ### nb match joue meme int/ext player2
            data_past_player1.loc[(data_past_player1['Winner'] == x[0]) &(data_past_player2['Court'] == x[6])].shape[0], ### nb match gagne meme int/ext player1 
            data_past_player2.loc[(data_past_player2['Winner'] == x[1]) &(data_past_player2['Court'] == x[6])].shape[0], ### nb match gagne meme int/ext player2
            
            data_past_players.shape[0], # nbr match joues ensemble
            data_past_players.loc[data_past_players["Winner"] == x[0]].shape[0], # nbr victoire joueur 1 sur joueur 2
            
            data_past_players.loc[data_past_players["Surface"] == x[2]].shape[0], # nbr match joues ensemble meme surface
            data_past_players.loc[(data_past_players["Surface"] == x[2])&(data_past_players["Winner"] == x[0])].shape[0], # nbr victoire joueur 1 sur joueur 2 meme surface
            
            ### nbr jeu moyen premier set quand perd
            ### nbr jeu moyen second set quand perd
            
            ### nombre tie break joues
            ### nbr tie break gagnes
        
            ### delta moyen de jeu premier set
            ### delta moyen de jeu second set
            
            )
#            
#            sum(((sub_data["Winner"] == x[i])&(sub_data["Court"] == 0))*1) ### nbr match won outdoor
            
            ### info manquante :
            
            ### index retour 
            ### index service 
            ### index blessure 
            ### index motivation
            ### index physique / athletic ou juste technique
    
    return stat



def data_prep_history(dataset):
    
    data = dataset.copy()
    ### integer rounds and calculate number of rounds
    
    ### dummify court
    data.loc[data["indoor_flag"] == "Outdoor", "indoor_flag"] = "0"
    data.loc[data["indoor_flag"] == "Indoor", "indoor_flag"]  = "1"
    data["Court"] = data["Court"].astype(int)
    
    #### date into days
    data["day_week"] = data["Date"].dt.dayofweek
    data["month"] = data["Date"].dt.month
    data["year"] = data["Date"].dt.year
    data["week"] = data["Date"].dt.week
    data["day_of_year"] = data["Date"].dt.dayofyear
    data["day_of_month"] = data["Date"].dt.day
    
    ### take care of empty set score
    for i in range(1,6):
        for k in ["L", "W"]:
            data[k + str(i)] = data[k + str(i)].replace(" ",0)
            data[k + str(i)] = data[k + str(i)].fillna(0) 

    columns = []
    columns += []
            
    t0 = time.time()        
    print("Start parallelisation")
    counts = parallelize_dataframe(data.copy(), create_stats, [data], 7)
    counts = list(zip(*counts))
    counts = np.transpose(counts)
    print(time.time() - t0)
    
    for i,col in enumerate(columns):
        data[col] = counts[:,i]
        
    #### nbr de fois deja gagne ce tournois
    #### nbr de fois deja franchis cette etape dans tournois
    
    data1 = data.copy()
    data1["target"] = 1
    
    ### take care of missing rank values
    data2 = data.copy()
    data2["target"] = 0
    
    data2.rename(columns = {'Winner': 'Loser', 'Loser': 'Winner', "elo1": "elo2", "elo2":"elo1"}, inplace = True)
         
    data2 = data2[data1.columns]
    data_concat = pd.concat([data1, data2], axis=0).sort_index().reset_index(drop=True)
    
    return data_concat

if __name__ == "__main__":
    data2 = data_prep_history(data)
    

