# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:14:40 2018

@author: JARD
"""

import pandas as pd
import glob
import numpy as np
from multiprocessing import Pool
from functools import partial
from datetime import datetime, timedelta
from sklearn.model_selection import cross_val_score
from sklearn import metrics, ensemble, linear_model, svm
from sklearn.metrics import roc_auc_score, accuracy_score
import time

from matplotlib import pyplot as plt


def merge_with_tournois(data_concat, path_tournament):
    
    tournois = pd.read_csv(path_tournament + "tournaments.csv", encoding = "latin1")[["Tournament", 'Prize', 'Surface', 'Indoor_flag',
       'Currency', 'ID', 'pays', 'City', "Date_start_tournament"]]
    tournois["Date_start_tournament"] = pd.to_datetime(tournois["Date_start_tournament"], format = "%d/%m/%Y")
    tournois["year"] = (tournois["Date_start_tournament"] +  timedelta(days=7)).dt.year
    tournois["key"] = tournois["Tournament"].astype(str) + "-" + tournois["year"].astype(str) 
    
    data_concat["Tournament"] = data_concat["Tournament"].apply(lambda x: x.replace("’", " "))
    data_concat["year"] = (data_concat["Date"] +  timedelta(days=7)).dt.year
    data_concat["key"] = data_concat["Tournament"].astype(str) + "-" + data_concat["year"].astype(str) 
    
    data_tournois = pd.merge(data_concat, tournois, on = ["key"], how = "left")
    
    data_tournois["nbr_days_since_tournament_start"] = (data_tournois["Date"] - data_tournois["Date_start_tournament"]).dt.days

    data_tournois.drop(["Surface_y", "key", "year_x", "year_y", "Tournament_y", "Indoor_flag"], axis= 1, inplace = True)
    
    return data_tournois


    