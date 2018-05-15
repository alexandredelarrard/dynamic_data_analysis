# -*- coding: utf-8 -*-
"""
Created on Fri May 11 14:51:18 2018

@author: JARD
"""

import time
import pandas as pd
import os

from create_models.data_prep.filter_data import data_prep_for_modelling
from create_models.models.xgb import modelling_xgboost
from create_models.models.logistic import modelling_logistic

def main_modelling(params):
    
    t0 = time.time()

    data =pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/overall/total_dataset_modelling.csv")
    
    ### data prep
    modelling_data = data_prep_for_modelling(data)
    modelling_data.to_csv(os.environ["DATA_PATH"]  + "/clean_datasets/modelling/data_modelling_V1.csv")
    
    ### modelling _ xgb
    clf, var_imp = modelling_xgboost(modelling_data, params["date_test_start"], params["date_test_end"])
    
    ### modelling logistic
#    clf, var_imp = modelling_xgboost(modelling_data, date_test_start = "2017-01-01", date_test_end="2017-12-31")
    
    print("Total time for data preparation for modelling {0}".format(time.time() - t0))
    return clf, var_imp, modelling_data

if __name__ == "__main__":
    clf, var_imp, modelling_data = main_modelling(retrain=False)