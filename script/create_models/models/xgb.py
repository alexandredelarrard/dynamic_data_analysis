# -*- coding: utf-8 -*-
"""
Created on Wed May  9 23:00:16 2018

@author: User
"""
import pandas as pd
import numpy as np
from sklearn import metrics, ensemble, linear_model, svm
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, plot_importance
import os

import warnings
warnings.filterwarnings("ignore")

from pylab import rcParams
rcParams['figure.figsize'] = (8, 12)

def modelling_xgboost(data, date_test_start, date_test_end):
     
    train, test_tot = split_train_test(data, date_test_start, date_test_end)
    avg_auc = 0
    avg_acc = 0
    avg_log_loss = 0
    for i, tourney in enumerate(test_tot.sort_values("Date")["tourney_name"].unique()):
        test = test_tot.loc[test_tot["tourney_name"]== tourney]
        
        if i >0:
            train= pd.concat([train, addi_train], axis=0)
    
        x_train, y_train, x_test, y_test = train.drop(["target", "Date", "tourney_name"],axis=1), train["target"], test.drop(["target", "Date", "tourney_name"],axis=1), test["target"]
        
        clf = XGBClassifier(    
                    n_estimators=500,
                    max_depth=7,
                    objective="binary:logistic",
                    learning_rate=0.07, 
                    subsample=0.9,
                    colsample_bytree=0.8,
                    min_child_weight=4,
                    reg_alpha = 1,
                    reg_lambda = 1,
                    gamma =0,
                 )
        
        eval_set=[(x_train, y_train), (x_test, y_test)]
        clf.fit(x_train, y_train, 
                           eval_set=eval_set,
                           eval_metric="logloss",
                           early_stopping_rounds=35,
                           verbose=False
                         )
            
        preds=  clf.predict_proba(x_test)[:,1]
        auc = roc_auc_score(y_test,preds)
        accuracy = accuracy_score(y_test, clf.predict(x_test))
        print("[{0}] : [AUC] {1} / [Accuracy] {2} / [logloss] {3} /  [Match Nbr] {4}".format(tourney, auc, accuracy, log_loss(y_test, preds), len(x_test)))
        
        avg_auc += auc*len(x_test) 
        avg_acc += accuracy*len(x_test) 
        avg_log_loss +=log_loss(y_test, preds)*len(x_test)
        var_imp = pd.DataFrame(np.array(clf.feature_importances_), columns = ["importance"], index= x_train.columns).sort_values("importance")
        addi_train = test
        
        plot_importance(clf)
        
    print("_"*40)
    print("[AUC avg] {0} / [Accuracy avg] {1} / [logloss] {2} / [Match Nbr total] {3} ".format(avg_auc/len(test_tot), avg_acc/len(test_tot), avg_log_loss/len(test_tot), len(test_tot)))
    return clf, var_imp


def split_train_test(data, date_test_start, date_test_end):
    
    train = data.loc[(data["Date"] < date_test_start)]
    test =  data.loc[(data["Date"] >= date_test_start) & (data["Date"]< date_test_end)]
    
    return train, test


if __name__ == "__main__":
    
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"
    data0 = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/variables_for_modelling_V1.csv")
    data0 = data0.loc[(~pd.isnull(data0["diff_aces"]))&(data0["Common_matches"]>5)]
    data0["Date"]= pd.to_datetime(data0["Date"], format = "%Y-%m-%d")
#    data0 = data0.loc[~pd.isnull(data0["diff_bp"])]
    
    clf, var_imp = modelling_xgboost(data0, date_test_start = "2017-01-01", date_test_end="2017-12-31")
    

#[AUC avg] 0.8423372516654938 / [Accuracy avg] 0.7711612079965972 / [Match Nbr total] 4702  ---> 2016
#[AUC avg] 0.8299067731919562 / [Accuracy avg] 0.7604046242774567 / [Match Nbr total] 3460  ---> 2017