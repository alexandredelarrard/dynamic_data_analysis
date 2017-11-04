# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:09:52 2017

@author: alexandre
"""

import pandas as pd
import numpy as np
#from sklearn import svm
import xgboost as xgb
from cross_validation import gridSearch


def RMSE_error(yhat, y):
    y = np.exp(y.get_label())
    yhat = np.exp(yhat)
    return  "RMSE", np.sqrt(np.mean((y - yhat)**2))

def MAPE_error(yhat, y):
    y = np.exp(y.get_label())
    yhat = np.exp(yhat)
    return  "MAPE", np.mean(abs(y - yhat))


def GBM_regression(data, Y_label, k_fold = {}, params= None):   
        
    data =data.copy()
    params = params = {"objective": "reg:linear",
                      "booster" : "gbtree",
                      "eta": 0.01,
                      "max_depth": 7,
                      "subsample": 0.88,
                      "colsample_bytree": 0.9,
                      "silent": 1,
                      "seed": 1301,
                      "min_child_weight" : 2,
                      "n_estimator" : 1000
                      }
    
    if len(k_fold) ==0:
        k_fold = {"train" : {"0" : data.index[:int(len(data)*0.8)].tolist()}, "test" : {"0": data.index[int(len(data)*0.8):].tolist()}}
    
    if params:
        real_params = params
        grid_search_params = {}
        for key, value in params.iteritems():
            if np.size(value) >1:
                grid_search_params[key] = value
                
        if len(grid_search_params)>0:
            clf = xgb.train(params, xgb.DMatrix(data[:min(int(len(data)/2), 10000)].drop(Y_label, axis=1), label=data.loc[:min(int(len(data)/2), 10000), Y_label]), 1000, 
                    feval= RMSE_error, verbose_eval=False)
        
            best_estimate = gridSearch(data, Y_label, clf, grid_search_params)
            
            for key, value in best_estimate.iteritems():
                real_params[key] = value
    
    
    for i in range(len(k_fold["train"])): 
        
        #### split train test rotation
        print(i)
        train_i = data.iloc[k_fold["train"][str(i)]]
        test_i = data.iloc[k_fold["test"][str(i)]]
        
        ##### train database creation   
        y_train = np.log(train_i[Y_label])    
        train_i_M = xgb.DMatrix(train_i.drop([Y_label], axis=1), label=y_train)
            
        ##### test database creation    
        y_test = np.log(test_i[Y_label])  
        test_i_M = xgb.DMatrix(test_i.drop(Y_label, axis=1), label=y_test)
        
        watchlist = [(train_i_M, 'train'), (test_i_M, 'eval')]
        clf_pred_bis =xgb.train(real_params, train_i_M, params["n_estimator"], evals=watchlist, \
                    early_stopping_rounds=40, feval= RMSE_error, verbose_eval=False)
        
        prediction_bis = clf_pred_bis.predict(xgb.DMatrix(test_i.drop(Y_label, axis=1)))
        rmse = np.sqrt(np.mean((test_i[Y_label] - np.exp(prediction_bis))**2))
        mape_bis = np.mean(abs((test_i[Y_label] - np.exp(prediction_bis))  / test_i[Y_label]))   
        test_i["residuals"] = ((test_i[Y_label] - np.exp(prediction_bis))  / test_i[Y_label])
        
        print("Etape %i: MAPE : %f RMSE %f"%(i, mape_bis, rmse))
    
    return clf_pred_bis, test_i
