# -*- coding: utf-8 -*-
"""
Created on Fri May 11 14:50:01 2018

@author: JARD
"""

import os
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

from create_train.main_create_train import main_create_train
from create_test.main_create_test import main_create_test
from create_models.main_modelling import main_modelling, main_prediction
#from create_finance.main_finance import main_finance

os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"


def global_main(params): 
    
    main_create_train(params)

    ### create test dataset after updating everything in train
    main_create_test(crawl_test = params["make_test"])
    
    #### model from the dataset updated from test creation
    now = datetime.now()
    params = {
              "date_test_start" :  now - timedelta(days = 0), 
              "date_test_end"   : now.strftime("%Y-%m-%d")
             }
    clf, var_imp, predictions_overall_xgb, predictions_overall_lg = main_modelling(params)
    
    #### predict the futur data
    predictions_futures = main_prediction()
    
    return predictions_futures, var_imp, predictions_overall_xgb, predictions_overall_lg


if __name__ == "__main__":

    params= {"rebuild" : True,
             "update": True,
             "make_test" : True}   
    
    predictions_futures, var_imp, predictions_overall_xgb, predictions_overall_lg = global_main(params)
    
    
    