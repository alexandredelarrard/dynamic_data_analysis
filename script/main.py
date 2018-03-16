# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:16:28 2018

@author: JARD
"""

import pandas as pd
import os
from modelling.modelling_lr import modelling_logistic
from data_prep.create_tournament import merge_with_tournois
from data_prep.exctract_data1 import import_data
from data_prep.create_statistics_history import data_prep_history
from data_prep.create_elo_ranking import merge_data_elo
from utils.plot_lib import var_vs_target


if __name__ == "__main__":
    
    ### read data and clean it 
    path = r"D:\projects\tennis betting\data\brute_info\historical\brute_info_origin"
    data = import_data(path)
    
    ### add elo system ranking
    data_elo = merge_data_elo(data)
    
    ### extract statistics and create target variable
#    data2 = data_prep_history(data_elo)
    
#    ### attatch tournament information
#    data3 = merge_with_tournois(data2)

#    #### lr modelling 
    clf, importance = modelling_logistic(data_elo, "2017-01-01", "2017-06-01", "gbm")
 