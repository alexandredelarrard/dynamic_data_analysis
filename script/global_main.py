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
from create_finance.main_finance import main_finance

os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"

if __name__ == "__main__":
    
    rebuild = {"redo_missing_atp_statistics" : True,
                   "create_elo" : True,
                   "create_variable" : True,
                   "create_statistics": True
                   }    
    
    full_data, modelling_data = main_creation(rebuild=rebuild)

