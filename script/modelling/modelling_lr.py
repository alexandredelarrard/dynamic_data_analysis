# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:15:24 2018

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
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, plot_importance


def modelling_logistic(data, date_test_start, date_test_end, method= "logistic"):
    
    print("fitting %s ..."%method)
    train, test_tot = split_train_test(data, date_test_start, date_test_end)
    avg_auc = 0
    avg_acc = 0
    
    for i, id_atp in enumerate(np.sort(test_tot["ATP"].unique())):
        test = test_tot.loc[test_tot["ATP"]== id_atp]
        
        if i >0:
            train= pd.concat([train, addi_train], axis=0)
    
        x_train, y_train, x_test, y_test = train.drop(["target", "Date"],axis=1), train["target"], test.drop(["target", "Date"],axis=1), test["target"]
        
        if method == "logistic":
            clf = linear_model.LogisticRegression(C=1)
            clf.fit(x_train, y_train)
            
            
        if method == "gbm":
            clf = XGBClassifier(    
                        n_estimators=500,
                        max_depth=6,
                        objective="binary:logistic",
                        learning_rate=0.08, 
                        subsample=0.8,
                        colsample_bytree=0.68,
                        min_child_weight=3,
                     )
            
            eval_set=[(x_test, y_test)]
            clf.fit(x_train, y_train, 
                               eval_set=eval_set,
                               eval_metric="logloss",
                               early_stopping_rounds=35,
                               verbose=False
                             )
            
        preds=  clf.predict_proba(x_test)[:,1]
        auc = roc_auc_score(y_test,preds)
        accuracy = accuracy_score(y_test, clf.predict(x_test))
        print("[ATP {0}] : [AUC] {1} / [Accuracy] {2} / [Match Nbr] {3}".format(id_atp, auc, accuracy, len(x_test)))
        
        avg_auc += auc*len(x_test) 
        avg_acc += accuracy*len(x_test) 
        
        if method == "logistic":
            scaler = StandardScaler(with_mean=False)
            scaler.fit(x_train)
            var_imp = pd.DataFrame(index=x_train.columns)
            
            var_imp["coefs"] =  abs(np.sqrt(scaler.var_)*clf.coef_[0])
            var_imp.sort_values(by = "coefs")
            
        else:
            var_imp = pd.DataFrame(np.array(clf.feature_importances_), columns = ["importance"], index= x_train.columns).sort_values("importance")
        
        addi_train = test

    print("_"*40)
    print("[ATP Nbr = {0}] : [AUC avg] {1} / [Accuracy avg] {2} / [Match Nbr total] {3}".format(len(test_tot["ATP"].unique()), avg_auc/len(test_tot), avg_acc/len(test_tot), len(test_tot)))
    return clf, var_imp


def split_train_test(data, date_test_start, date_test_end):
    
    train = data.loc[(data["Date"] < date_test_start) & (data["Date"]>="2001-01-01")]
    test =  data.loc[(data["Date"] >= date_test_start) & (data["Date"]< date_test_end)]
    
    return train, test
