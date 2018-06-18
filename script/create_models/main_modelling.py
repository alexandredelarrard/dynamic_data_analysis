# -*- coding: utf-8 -*-
"""
Created on Fri May 11 14:51:18 2018

@author: JARD
"""

import sys
import pandas as pd
import os
from datetime import datetime
import pickle as pkl

sys.path.append(r"C:\Users\User\Documents\tennis\dynamic_data_analysis\script")
from create_models.data_prep.filter_data import data_prep_for_modelling
from create_models.models.xgb import modelling_xgboost
from create_models.models.logistic import modelling_logistic
from create_train.data_update.main_create_update import import_data

def main_prediction():
    
    cols_used = ['Common_matches', 'best_of', 'bst_rk_l', 'bst_rk_w', 'day_week', 'day_of_year', 'diff_1st_serv_in', 'diff_1st_serv_won', 'diff_2nd_serv_won', 'diff_aces', 'diff_age', 'diff_bp', 'diff_df', 'diff_elo', 'diff_fatigue_games', 'diff_hand', 'diff_home', 'diff_ht', 'diff_imc', 'diff_mean_rank_adversaries', 'diff_overall_skill', 'diff_pts_common_matches', 'diff_rank', 'diff_rk_pts', 'diff_serv1_ret2', 'diff_serv2_ret1', 'diff_skill_ret', 'diff_skill_serv', 'diff_tie_break', 'diff_victories_common_matches', 'diff_weight', 'diff_weights', 'diff_year_turned_pro', 'draw_size', 'round', 'nbr_reach_level_tourney_l', 'diff_victories_12', 'nbr_reach_level_tourney_w', 'prize', 'prop_last_set_gagne_l', 'prop_last_set_gagne_w', 'prop_victory_surface_l', 'prop_victory_surface_w', 'winner_rank', 'winner_rank_points', 'loser_rank', 'loser_rank_points', 'w_birthday', 'l_birthday', 'diff_time_set', 'masters_1000', 'masters_1000s', 'masters_250', 'masters_500', 'masters_classic', 'masters_finals.svg', 'masters_grandslam', 'surface_Carpet_1', 'surface_Clay_0', 'surface_Grass_0', 'surface_Hard_0', 'surface_Hard_1']
    
    test_data = pd.read_csv(os.environ["DATA_PATH"] + "/test/test_{0}.csv".format(datetime.now().strftime("%Y-%m-%d")))
    test_data["Date"] = pd.to_datetime(test_data["Date"], format = "%Y-%m-%d")
    
    preds = pd.DataFrame([])
    preds[["Date", 'tourney_name', 'winner_name', 'loser_name']] = test_data[["Date", 'tourney_name', 'winner_name', 'loser_name']]
    
    #### prepare the test set
    test_data_futur = data_prep_for_modelling(test_data)
    preds = preds.loc[test_data_futur.index]
    
    cols_missing = [x for x in cols_used if x not in test_data_futur.columns]
    print(" Adding following cols to test set as they are missing {0}".format(cols_missing))
    for col in cols_missing :
        test_data_futur[col] = 0
    test_data_futur = test_data_futur[cols_used]
    
    #### predict with xgb 
    xgb_clf = pkl.load(open(r"C:\Users\User\Documents\tennis\models\match_proba_prediction\xgb\xgb_{0}.pkl".format(datetime.now().strftime("%Y-%m-%d")), 'rb'))
    preds["xgb_pred"] = xgb_clf.predict_proba(test_data_futur)[:,1]
    
    #### predict with logistic 
    logit_clf = pkl.load(open(r"C:\Users\User\Documents\tennis\models\match_proba_prediction\logistic\logistic_{0}.pkl".format(datetime.now().strftime("%Y-%m-%d")), 'rb'))
    preds["logit_pred"] = logit_clf.predict_proba(test_data_futur)[:,1]

    return preds
    

def main_modelling(params):
    
    data = import_data()
    
    ### data prep
    modelling_data = data_prep_for_modelling(data)
    
    ### modelling logistic
    clf, var_imp, predictions_overall_lg = modelling_logistic(modelling_data, params["date_test_start"], params["date_test_end"])
    
    ### modelling _ xgb
    clf, var_imp, predictions_overall_xgb = modelling_xgboost(modelling_data, params["date_test_start"], params["date_test_end"])

    return clf, var_imp, predictions_overall_xgb, predictions_overall_lg

if __name__ == "__main__":
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"
    params = {
            "date_test_start" : "2017-05-01", 
            "date_test_end"   : "2018-06-13"
             }
    clf, var_imp, predictions_overall_xgb, predictions_overall_lg = main_modelling(params)