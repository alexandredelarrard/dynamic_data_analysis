# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:15:24 2018

@author: JARD
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import os

import warnings
warnings.filterwarnings("ignore")


def modelling_logistic(data, date_test_start, date_test_end):
    
    data = data[["target", "Date", 'diff_elo','diff_rank', "tourney_name", 'diff_rk_pts', 'diff_hand', 'diff_is_birthday',
                 'diff_home', 'diff_age', 'diff_ht', 'diff_days_since_stop', 'Common_matches', "diff_fatigue_games",
                 'diff_aces', 'diff_df', 'diff_1st_serv_in','diff_1st_serv_won', 'diff_2nd_serv_won', 'diff_skill_serv',
                   'diff_skill_ret', 'diff_overall_skill', 'diff_serv1_ret2', 'diff_victories_common_matches', 
                   'diff_serv2_ret1', 'diff_bp', 'diff_weight', 'diff_year_turned_pro', 'diff_tie_break', 'diff_victories_12',]]
        
    
    train, test_tot = split_train_test(data, date_test_start, date_test_end)
    avg_auc = 0
    avg_acc = 0
    
    for i, tourney in enumerate(np.sort(test_tot["tourney_name"].unique())):
        test = test_tot.loc[test_tot["tourney_name"]== tourney]
        
        if i >0:
            train= pd.concat([train, addi_train], axis=0)
    
        x_train, y_train, x_test, y_test = train.drop(["target", "Date", "tourney_name"],axis=1), train["target"], test.drop(["target", "Date", "tourney_name"],axis=1), test["target"]
        
        clf = linear_model.LogisticRegression(C=1)
        clf.fit(x_train, y_train)
    
        preds=  clf.predict_proba(x_test)[:,1]
        auc = roc_auc_score(y_test,preds)
        accuracy = accuracy_score(y_test, clf.predict(x_test))
        print("[{0}] : [AUC] {1} / [Accuracy] {2} / [Match Nbr] {3}".format(tourney, auc, accuracy, len(x_test)))
        
        avg_auc += auc*len(x_test) 
        avg_acc += accuracy*len(x_test) 
        
        scaler = StandardScaler(with_mean=False)
        scaler.fit(x_train)
        var_imp = pd.DataFrame(index=x_train.columns)
        
        var_imp["coefs"] =  abs(np.sqrt(scaler.var_)*clf.coef_[0])
        var_imp.sort_values(by = "coefs")
            
        addi_train = test

    print("_"*40)
    print("[AUC avg] {0} / [Accuracy avg] {1} / [Match Nbr total] {2}".format(avg_auc/len(test_tot), avg_acc/len(test_tot), len(test_tot)))
    return clf, var_imp


def split_train_test(data, date_test_start, date_test_end):
    
    train = data.loc[(data["Date"] < date_test_start)]
    test =  data.loc[(data["Date"] >= date_test_start) & (data["Date"]< date_test_end)]
    
    return train, test


if __name__ == "__main__":
    
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"
    data0 = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/variables_for_modelling_V1.csv")
    data0["Date"]= pd.to_datetime(data0["Date"], format = "%Y-%m-%d")
    
    data0 = data0.loc[(~pd.isnull(data0["diff_aces"]))&(data0["Common_matches"]>5)]
    data0 = data0.loc[~data0["diff_bp"].isin([np.nan, np.Inf, -np.Inf])]
    
    clf, var_imp = modelling_logistic(data0, date_test_start = "2016-01-01", date_test_end="2016-03-31")
    
