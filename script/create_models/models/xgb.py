# -*- coding: utf-8 -*-
"""
Created on Wed May  9 23:00:16 2018

@author: User
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from xgboost import XGBClassifier, plot_importance
import os
import pickle as pkl
from datetime import datetime

from pylab import rcParams
rcParams['figure.figsize'] = (8, 12)


def modelling_xgboost(data, date_test_start, date_test_end):
     
    train, test_tot = split_train_test(data, date_test_start, date_test_end)
    avg_auc = 0
    avg_acc = 0
    avg_log_loss = 0
    print("\n")
    
    params = {"n_estimators":2000,
             "max_depth":6,
             "objective":"binary:logistic",
             "learning_rate":0.045, 
             "subsample":0.9,
             "colsample_bytree":0.85,
             "min_child_weight":2,
             "n_jobs":8,
             "reg_alpha": 1.07,
             "reg_lambda": 1,
             "gamma":0
            }
    
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
        
        clf = XGBClassifier(**params)

        eval_set=[(x_train, y_train), (x_test, y_test)]
        clf.fit(x_train, y_train, 
                           eval_set=eval_set,
                           eval_metric="logloss",
                           early_stopping_rounds=40,
                           verbose=False
                         )
            
        preds=  clf.predict_proba(x_test)[:,1]
        
        if i ==0:
            predictions =  pd.DataFrame(preds, columns= ["preds"])
        else:
            predictions = pd.concat([predictions, pd.DataFrame(preds)], axis=0)
        
        auc = roc_auc_score(y_test,preds)
        accuracy = accuracy_score(y_test, clf.predict(x_test))
        print("[{0: <16}][{5}] : [AUC] {1:.3f} / [Accuracy] {2:.3f} / [logloss] {3:.3f} /  [Match Nbr] {4}".format(tourney, auc, accuracy, log_loss(y_test, preds), len(x_test), date_ref))
        
        avg_auc += auc*len(x_test) 
        avg_acc += accuracy*len(x_test) 
        avg_log_loss +=log_loss(y_test, preds)*len(x_test)
        var_imp = pd.DataFrame(np.array(clf.feature_importances_), columns = ["importance"], index= x_train.columns).sort_values("importance")
        addi_train = test
        
    if test_tot.shape[0]>0:
        plot_importance(clf)
        predictions_overall["preds"] = predictions[0].tolist()
        print("_"*40)
        print("[AUC avg] {0} / [Accuracy avg] {1:.3f} / [logloss] {2:.3f} / [Match Nbr total] {3} ".format(avg_auc/len(test_tot), avg_acc/len(test_tot), avg_log_loss/len(test_tot), len(test_tot)))

    else:
         predictions_overall = ""
         var_imp = ""
    
    ### train model on all data 
    print("\n fitting the overall dataset for real prediction")
    clf = XGBClassifier(**params)
    clf.fit(data.drop(["target", "Date", "tourney_name"], axis=1), data["target"])
    pkl.dump(clf, open(r"C:\Users\User\Documents\tennis\models\match_proba_prediction\xgb\xgb_{0}.pkl".format(datetime.now().strftime("%Y-%m-%d")), "wb"))
    
    return clf, var_imp, predictions_overall


def modelling_xgboost_tuning(data, date_test_start, date_test_end, params= False):
    
    train, test_tot = split_train_test(data, date_test_start, date_test_end)
    avg_auc = 0
    avg_acc = 0
    avg_log_loss = 0
    print("\n")
    
    if not params:
        params = {"n_estimators":2000,
             "max_depth":6,
             "objective":"binary:logistic",
             "learning_rate":0.045, 
             "subsample":0.9,
             "colsample_bytree":0.85,
             "min_child_weight":2,
             "n_jobs":8,
             "reg_alpha": 1.05,
             "reg_lambda": 1,
            }
    
    
    test = test_tot
    predictions_overall = test
    x_train, y_train, x_test, y_test = train.drop(["target", "Date", "tourney_name"],axis=1), train["target"], test.drop(["target", "Date", "tourney_name"],axis=1), test["target"]
    
    clf = XGBClassifier(**params)

    eval_set=[(x_train, y_train), (x_test, y_test)]
    clf.fit(x_train, y_train, 
                       eval_set=eval_set,
                       eval_metric="logloss",
                       early_stopping_rounds=40,
                       verbose=False
                     )
        
    preds=  clf.predict_proba(x_test)[:,1]

    auc = roc_auc_score(y_test,preds)
    accuracy = accuracy_score(y_test, clf.predict(x_test))
    print("[AUC] {1:.3f} / [Accuracy] {2:.3f} / [logloss] {3:.3f} /  [Match Nbr] {4}".format(0, auc, accuracy, log_loss(y_test, preds), len(x_test)))
    
    avg_auc += auc*len(x_test) 
    avg_acc += accuracy*len(x_test) 
    avg_log_loss +=log_loss(y_test, preds)*len(x_test)
    var_imp = pd.DataFrame(np.array(clf.feature_importances_), columns = ["importance"], index= x_train.columns).sort_values("importance")
        
    plot_importance(clf)
    predictions_overall["preds"] = preds
  
    return clf, var_imp, predictions_overall


def split_train_test(data, date_test_start, date_test_end):
    
    date_test_start = pd.to_datetime(date_test_start, format = "%Y-%m-%d")
    date_test_end   = pd.to_datetime(date_test_end, format = "%Y-%m-%d")
    
    train = data.loc[(data["Date"] < date_test_start)]
    test =  data.loc[(data["Date"] >= date_test_start) & (data["Date"]< date_test_end)]
    
    return train, test


if __name__ == "__main__":
    
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"
    data0 = pd.read_csv(os.environ["DATA_PATH"]  + "/clean_datasets/historical/variables_for_modelling_V1.csv")
    data0 = data0.loc[(~pd.isnull(data0["diff_aces"]))&(data0["Common_matches"]>5)]
    data0["Date"]= pd.to_datetime(data0["Date"], format = "%Y-%m-%d")
    
    clf, var_imp, predictions_overall = modelling_xgboost(data0, date_test_start = "2017-01-01", date_test_end="2017-12-31")
    