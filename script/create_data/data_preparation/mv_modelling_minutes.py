# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 11:06:33 2018

@author: User
"""

import pandas as pd
import xgboost as xgb
import os
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (8,15)
from prepare_data_modelling_mvs import import_data_atp

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def minutes_modelling(data, test, Y_label, params):
    
    keep = ["month", "year", "day_of_year", "day_of_month", 'round_F', 'round_QF',
           'round_R128', 'round_R16', 'round_R32', 'round_R64', 'round_RR',
           'round_SF', 'surface_Carpet_1', 'surface_Clay_0', 'surface_Grass_0',
           'surface_Hard_0', 'surface_Hard_1', 'draw_size', 'winner_hand',
           'winner_ht', 'winner_age', 'winner_rank', 'winner_rank_points',
           'loser_hand', 'loser_ht', 'loser_age', 'loser_rank',
           'loser_rank_points', 'minutes', 'indoor_flag', 
           'w_imc', 'l_imc', 'total_games',
           'N_set', 'w_S1', 'l_S1', 'w_S2', 'l_S2', 'w_S3', 'l_S3', 'w_S4', 'l_S4',
           'w_S5', 'l_S5', 'total_tie_break_w', 'total_tie_break_l']
    
    sample_submission = pd.DataFrame(test.index)
    sample_submission['minutes'] = 0
    
    data = data[keep]
    test = test[keep]
    data = data.sample(frac=1).reset_index(drop=True)
    
    data2 = data.copy()
    
    for col in [x for x in data2.columns if "w_" == x[:2]]:
        data2.rename(columns={col : col.replace("w_","l_"), col.replace("w_","l_") : col}, inplace = True)
        
    for col in [x for x in data2.columns if "_w" == x[-2:]]:
        data2.rename(columns={col : col.replace("_w","_l"), col.replace("_w","_l") : col}, inplace = True)
    
    for col in [x for x in data2.columns if "winner_" in x]:
        data2.rename(columns={col : col.replace("winner_","loser_"), col.replace("winner_","loser_") : col}, inplace = True)
    
    data = pd.concat([data, data2], axis=0)
    data = data.sort_index().reset_index(drop=True)
    
    k_fold = params["k_fold"]

    avg_mape = []
    avg_rmse = []
    avg_mae = []
    
    dataset_importance = pd.DataFrame([], columns = ["variables", "average_importance"])
    dataset_importance["variables"] = data.drop(Y_label, axis= 1).columns
    
    X = data.drop(Y_label, axis= 1)
    y = np.log(data[Y_label])
   
    Kf = KFold(n_splits=k_fold, shuffle= False, random_state= 36520)

    for i, (train_i, test_i) in enumerate(Kf.split(X, y)):
        
        X_train, X_test = X.loc[train_i], X.loc[test_i]
        y_train, y_test = y.loc[train_i], y.loc[test_i]
        
        eval_set  = [(X_train,y_train), (X_test,y_test)]
        
        clf = xgb.XGBRegressor(**params)
        clf.fit(X_train, y_train, 
                eval_set=eval_set,
                eval_metric="rmse",
                early_stopping_rounds=40,
                verbose= 0)
        
        dataset_importance["importance_%i"%i] = clf.feature_importances_
        
        preds = clf.predict(X_test)
        pp = pd.DataFrame(np.transpose([np.exp(y_test).tolist(), np.exp(preds).tolist()]), columns = ["true", "pred"])
    
        print("[Fold {0}] MAPE : {1:.2f}, RMSE : {2:.2f}, MAE {3:.2f}".format(i ,(abs(pp["true"] - pp["pred"])*100/pp["true"]).mean(), np.sqrt(mean_squared_error(pp["true"], pp["pred"]) ), abs(pp["true"] - pp["pred"]).mean()  ))
        avg_mape.append((abs(pp["true"] - pp["pred"])*100/pp["true"]).mean())
        avg_rmse.append(np.sqrt(mean_squared_error(pp["true"], pp["pred"]) ))
        avg_mae.append(abs(pp["true"] - pp["pred"]).mean())
        
#        sample_submission['minutes'] += np.exp(clf.predict(test.drop("minutes",axis=1))) #### predict for the submission set : test
         
    print("_"*40)
    print("[OVERALL] MAPE : {0:.2f}, RMSE : {1:.2f}, MAE {2:.2f}".format(np.mean(avg_mape), np.mean(avg_rmse), np.mean(avg_mae)))
    
    xgb.plot_importance(clf)
    
    dataset_importance["average_importance"] = dataset_importance[["importance_%i"%i for i in range(k_fold)]].mean(axis=1)
  
    return clf, pp, dataset_importance.sort_values("importance"), sample_submission


def results_analysis(preds):
    
    fig, ax = plt.subplots()

    ax.scatter(preds["true"], preds["pred"], alpha = 0.5, s= 5, label= ["true", "pred"])
    lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, color= "red")
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.xlabel('true', fontsize=16)
    plt.ylabel('prediction', fontsize=16)
    

if __name__ == "__main__":
    os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"
    path = os.environ["DATA_PATH"]  + "/brute_info/historical/brute_info_atp/"
    train, test, train_minutes, test_minutes = import_data_atp(path)
    
    for lr in [0.03,0.04,0.05,0.06,0.07]:
        params = {"objective" : 'reg:linear',
                  "n_estimators": 2000, 
                  "learning_rate": lr,
                  "subsample": 0.8,
                  "colsample_bytree":0.8,
                  "max_depth":4,
                  "gamma":0, 
                  "reg_alpha" :1,
                  "reg_lambda" : 1,
                  "min_child_weight":3,
                  "seed" : 7666,
                  "k_fold": 5}
        
        model, results, importance, test_predictions = minutes_modelling(train_minutes, test_minutes, Y_label = "minutes", params= params)
        results_analysis(results)
        