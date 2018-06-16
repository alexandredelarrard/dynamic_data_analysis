# -*- coding: utf-8 -*-
"""
Created on Fri May 11 13:27:50 2018                    

@author: JARD
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (8,15)

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

### utils
from utils.utils_dataprep import data_prep
from utils.plot_lib import results_analysis

def select_train_test(data, Y_label):
    
    if Y_label == "minutes":
        data["time/games"] = data["minutes"]/data["total_games"]
        data = data.loc[data["time/games"].between(2,7)]
        data = data[["minutes", 'best_of', 'winner_age', 'loser_age', 'winner_ht', 'loser_ht', 'surface', 'indoor_flag', 'N_set','w_svpt', 
                     'masters', 'elo1', 'elo2', 'month', 'total_games', 'draw_size', 'Nbr_tie-breaks', 'l_svpt','l_bpFaced','w_bpFaced',
                     "winner_rank", "winner_rank_points", 'loser_rank', 'loser_rank_points',"day_of_year", "week"]]

    return data


def minutes_modelling(data, Y_label, params):
    
    data = data.reset_index(drop=True).copy()
    
    k_fold = params["k_fold"]

    avg_mape = []
    avg_rmse = []
    avg_mae = []
    
    dataset_importance = pd.DataFrame([], columns = ["variables", "importance"])
    dataset_importance["variables"] = data.drop(Y_label, axis= 1).columns
    
    X = data.drop(Y_label, axis= 1)
    y = np.log(data[Y_label])
   
    Kf = KFold(n_splits=k_fold, shuffle= True, random_state= 36520)

    for i, (train_i, test_i) in enumerate(Kf.split(X, y)):
        
        X_train, X_test = X.loc[train_i], X.loc[test_i]
        y_train, y_test = y.loc[train_i], y.loc[test_i]
        
        eval_set  = [(X_train,y_train), (X_test,y_test)]
        
        clf = xgb.XGBRegressor(**params)
        clf.fit(X_train, y_train, 
                eval_set=eval_set,
                eval_metric="rmse",
                early_stopping_rounds=30,
                verbose= 0)
        
        dataset_importance["importance_%i"%i] = clf.feature_importances_
        
        preds = clf.predict(X_test)
        pp = pd.DataFrame(np.transpose([np.exp(y_test).tolist(), np.exp(preds).tolist()]), columns = ["true", "pred"])
    
        print("[Fold {0}] MAPE : {1}, RMSE : {2}, MAE {3}".format(i ,(abs(pp["true"] - pp["pred"])*100/pp["true"]).mean(), np.sqrt(mean_squared_error(pp["true"], pp["pred"]) ), abs(pp["true"] - pp["pred"]).mean()  ))
        avg_mape.append((abs(pp["true"] - pp["pred"])*100/pp["true"]).mean())
        avg_rmse.append(np.sqrt(mean_squared_error(pp["true"], pp["pred"]) ))
        avg_mae.append(abs(pp["true"] - pp["pred"]).mean())
         
    print("_"*40)
    print("[OVERALL] MAPE : {0}, RMSE : {1}, MAE {2}".format(np.mean(avg_mape), np.mean(avg_rmse), np.mean(avg_mae)))
    
    xgb.plot_importance(clf)
    
    dataset_importance["average_importance"] = dataset_importance[["importance_%i"%i for i in range(k_fold)]].mean(axis=1)
  
    return clf, pp, dataset_importance.sort_values("average_importance")[["variables", "average_importance"]]


if __name__ == "__main__":
    
    path = r"C:\Users\JARD\Documents\projects\tennis\data\total_dataset_modelling.csv"
    data = pd.read_csv(path)
#    
    params = {"objective" : 'reg:linear',
              "n_estimators": 2000, 
              "learning_rate": 0.04,
              "subsample": 0.85,
              "colsample_bytree":0.85,
              "max_depth":5,
              "gamma":0, 
              "reg_alpha" :1.1,
              "reg_lambda" : 1.1,
              "min_child_weight":4,
              "seed" : 7666,
              "k_fold": 5}
    
    data2 = select_train_test(data, Y_label= "minutes")
    data3 = data_prep(data2)
    model, results, importance = minutes_modelling(data3, Y_label = "minutes", params= params)
    results_analysis(results)
    
    
#[Fold 0] MAPE : 6.791750033145004, RMSE : 9.613748730100372, MAE 6.984112262446959
#[Fold 1] MAPE : 6.883092280645388, RMSE : 9.796427340462326, MAE 7.066586749665573
#[Fold 2] MAPE : 6.872883137542973, RMSE : 9.819295354463382, MAE 7.118024074178871
#[Fold 3] MAPE : 6.894246117653847, RMSE : 9.857573311776484, MAE 7.0947646706095275
#[Fold 4] MAPE : 6.818169756194742, RMSE : 10.313266366300065, MAE 7.062499063596976
#________________________________________
#[OVERALL] MAPE : 6.852028265036391, RMSE : 9.880062220620527, MAE 7.06519736409958