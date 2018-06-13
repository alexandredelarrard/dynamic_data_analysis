# -*- coding: utf-8 -*-
"""
Created on Fri May 11 14:51:18 2018

@author: JARD
"""

import sys

sys.path.append(r"C:\Users\User\Documents\tennis\dynamic_data_analysis\script\create_models")
from data_prep.filter_data import data_prep_for_modelling
from models.xgb import modelling_xgboost
from models.logistic import modelling_logistic

def main_modelling(params):
    
    ### data prep
    modelling_data = data_prep_for_modelling()
    
    ### modelling _ xgb
    clf, var_imp, predictions_overall_xgb = modelling_xgboost(modelling_data, params["date_test_start"], params["date_test_end"])
    
    ### modelling logistic
    clf, var_imp, predictions_overall_lg = modelling_logistic(modelling_data, params["date_test_start"], params["date_test_end"])

    return clf, var_imp, predictions_overall_xgb, predictions_overall_lg

if __name__ == "__main__":
    params = {
            "date_test_start" : "2017-05-01", 
            "date_test_end"   : "2018-06-13"
             }
    clf, var_imp, predictions_overall_xgb, predictions_overall_lg = main_modelling(params)