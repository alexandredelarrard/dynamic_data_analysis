# -*- coding: utf-8 -*-
"""
Created on Fri May 11 14:51:18 2018

@author: JARD
"""

import time
import pandas as pd
import os

from data_prep.filter_data import data_prep_for_modelling

def main_modelling(retrain=False):
    
    t0 = time.time()

    data =pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/overall/total_dataset_modelling.csv")
    modelling_data = data_prep_for_modelling(data)
    
    modelling_data.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/modelling/data_modelling.csv")
    
    print("Total time for data preparation for modelling {0}".format(time.time() - t0))
    return modelling_data
