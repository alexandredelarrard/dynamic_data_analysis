# -*- coding: utf-8 -*-
"""
Created on Fri May 11 14:50:01 2018

@author: JARD
"""

import os
import warnings
warnings.filterwarnings("ignore")

from create_data.main_create import main_creation
from create_models.main_modelling import main_modelling
#from create_finance.main_finance import main_finance

os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"

if __name__ == "__main__":
    
    #### create the dataset
    rebuild = {"redo_missing_atp_statistics" : False,
               "create_elo" : False,
               "create_variable" : False,
               "create_statistics": False,
               "update_data": True
              }    
    
    full_data = main_creation(rebuild=rebuild)
    
    #### model from the dataset
    params = {
            "date_test_start" : "2017-01-01", 
            "date_test_end":"2017-12-31"
             }
    
    clf, var_imp, modelling_data = main_modelling(params)
    