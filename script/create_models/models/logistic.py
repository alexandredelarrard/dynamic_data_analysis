# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:15:24 2018

@author: JARD
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
import os
import pickle as pkl
from datetime import datetime


def modelling_logistic(data, date_test_start, date_test_end):
     
    train, test_tot = split_train_test(data, date_test_start, date_test_end)
    avg_auc = 0
    avg_acc = 0
    avg_log_loss = 0
    print("\n")
    
    for i, tourney in enumerate(test_tot.sort_values("Date")["tourney_name"].unique()):
        test = test_tot.loc[test_tot["tourney_name"]== tourney]
        date_ref = test["Date"].min()
        
        if i == 0:
            predictions_overall = test
        else:
            predictions_overall = pd.concat([predictions_overall, test], axis=0).reset_index(drop=True)
        
        if i >0:
            train= pd.concat([train, addi_train], axis=0)
        
        x_train, y_train, x_test, y_test = train.drop(["target", "Date", "tourney_name"],axis=1), train["target"], test.drop(["target", "Date", "tourney_name"],axis=1), test["target"]
        
        clf = linear_model.LogisticRegression(C=1)
        clf.fit(x_train, y_train)
        preds = clf.predict_proba(x_test)[:,1]
        
        if i ==0:
            predictions =  pd.DataFrame(preds, columns= ["preds"])
        else:
            predictions = pd.concat([predictions, pd.DataFrame(preds)], axis=0)
            
        auc = roc_auc_score(y_test, preds)
        accuracy = accuracy_score(y_test, clf.predict(x_test))
        print("[{0: <16}][{5}] : [AUC] {1:.3f} / [Accuracy] {2:.3f} / [logloss] {3:.3f} /  [Match Nbr] {4}".format(tourney, auc, accuracy, log_loss(y_test, preds), len(x_test), date_ref))
        
        avg_auc += auc*len(x_test) 
        avg_acc += accuracy*len(x_test) 
        avg_log_loss +=log_loss(y_test, preds)*len(x_test)
        scaler = StandardScaler(with_mean=False)
        scaler.fit(x_train)
        var_imp = pd.DataFrame(index=x_train.columns)
        
        var_imp["coefs"] =  abs(np.sqrt(scaler.var_)*clf.coef_[0])
        var_imp.sort_values(by = "coefs")
        addi_train = test
    
    if test_tot.shape[0]>0:
        var_imp.plot(kind="bar", figsize = (15,10), rot = 85)
        predictions_overall["preds"] = predictions[0].tolist()
        print("_"*40)
        print("[AUC avg] {0} / [Accuracy avg] {1:.3f} / [logloss] {2:.3f} / [Match Nbr total] {3} ".format(avg_auc/len(test_tot), avg_acc/len(test_tot), avg_log_loss/len(test_tot), len(test_tot)))
    
    else:
         predictions_overall = pd.DataFrame([])
         var_imp = ""
    
    #### export model
    print("\n fitting the overall dataset for real prediction")
    clf = linear_model.LogisticRegression(C=1)
    clf.fit(data.drop(["target", "Date", "tourney_name"], axis=1), data["target"])
    pkl.dump(clf, open(r"C:\Users\User\Documents\tennis\models\match_proba_prediction\logistic\logistic_{0}.pkl".format(datetime.now().strftime("%Y-%m-%d")), "wb"))
    
    return clf, var_imp, predictions_overall


def split_train_test(data, date_test_start, date_test_end):
    
    date_test_start = pd.to_datetime(date_test_start, format = "%Y-%m-%d")
    date_test_end   = pd.to_datetime(date_test_end, format = "%Y-%m-%d")
    
    train = data.loc[(data["Date"] < date_test_start)]
    test =  data.loc[(data["Date"] >= date_test_start) & (data["Date"]< date_test_end)]
    
    return train, test


if __name__ == "__main__":
    
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"

    clf, var_imp, predictions_overall = modelling_logistic(data, date_test_start = "2016-01-01", date_test_end="2016-03-31")
    
