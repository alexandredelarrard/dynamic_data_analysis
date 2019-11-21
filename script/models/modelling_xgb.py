"""
Created on Thu May 17 12:41:18 2018

@author: JARD
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import time
import random
import sys
import pickle
import os

plt.rcParams["figure.figsize"] = (10,12)

import xgboost as xgb

sys.path.append(os.environ["original_path"] + "/dataanalytics/utils")
from script.plotting_library import auc_roc_curves_gbm_glm

def k_fold_split(X, k_fold):
    splits = {}
    unique = list(np.unique(X.index))
    new_unique = unique
    lengths = [len(unique)//k_fold]*(k_fold -1)  
    lengths += [len(unique) - sum(lengths)]
    for k in range(k_fold):
        splits[k] = {}
        splits[k]["Test"]  = random.sample(new_unique, lengths[k])
        splits[k]["Train"] = list(set(unique) - set(splits[k]["Test"]))
        new_unique = list(set(new_unique) - set(splits[k]["Test"]))
    return splits


# =============================================================================
# Train a XGB model on a K-Fold
# =============================================================================
def modelling_xgb(args, data, Y_label, params):
    '''
    This function takes the prepared data, the output Y_label and train 5 models based on a 80/20 train test split
    Dpending if this is a classification or a regression, the XGBClassifier or XGBRegressor will be called.
    - For each of the fold, KPI like AUC or logloss or rmse are saved into _folds dataframe. 
    - residuals dataframe will concatenate all necessary features and predictions for each 20% test set.
    - dataset_importance is saving feature importance for all folds
    
    Inputs: - arguments as a dictionnary going through all the framework
            - data: dataprep dataset with binarized categorical features
            - Y_label: output name to predict
            - params : dictionnary of the hyper parameters of the model
            
    Ouputs: - clf : 4th fold model
            - residuals: all predictions, also called preds into report program
            - dataset_importance : feature importance of the 4th fold
            - _folds: 5-fold KPI's useful for the report 
    '''
    
    print("\n" + "--"*10 + "  Modelling  %s"%args["insurer"] + "--"*10 +"\n")
    random.seed(params["seed"])
    data = data.copy() 
    
    X = data.drop(Y_label, axis= 1)

    print("Total shape to predict " + str(X.shape))
    
    # if the output can have values higher than 100, then the logarithm of the output is predicted
    # this rule enables to have a model in general more robust to outliers
    if max(data[Y_label])> 100:
        y = np.log(data[Y_label])
    else:
        y = data[Y_label]

    residuals = pd.DataFrame()
    dataset_importance = pd.DataFrame([], columns = ["variables", "importance"])
    dataset_importance["variables"] = X.drop(['sample_number'], axis=1).columns
    
    if params["objective"] == "binary:logistic" or params["objective"] == "multi:softmax":
        _folds = pd.DataFrame([], index = ["Fold_{0}".format(x) for x in range(params["k_fold"])] + ["Total"], columns = ["AUC", "logloss", "Deviance"])
    else:
        _folds = pd.DataFrame([], index = ["Fold_{0}".format(x) for x in range(params["k_fold"])] + ["Total"], columns = ["MAPE (%)", "RMSE", "MAE (AU$)"])
        
    print("_"*40)
    print("----> start to train the 5 fold xgboost on cover {0}".format(args["cover"]))
    
    index = []
    
    # loop for each fold, create a train test split on the available index of the X dataset (features without output)
    for i, value in k_fold_split(X, params["k_fold"]).items():
        X_train, X_test = X.loc[value["Train"]], X.loc[value["Test"]]
        y_train, y_test = y.loc[value["Train"]], y.loc[value["Test"]]
        
        index = np.concatenate((index, X_test.index), axis = 0)
        print("Train {0}, Test {1}, avg_train {2:.3f}, avg_test {3:.3f}".format(X_train.shape, X_test.shape, y_train.mean(), y_test.mean()))
        
        eval_set  = [(X_train.drop(['sample_number'], axis=1), y_train), (X_test.drop(['sample_number'], axis=1), y_test)]
        
        if params["objective"] == "binary:logistic" or params["objective"] == "multi:softmax":
            clf = xgb.XGBClassifier(**params)
        else:
            clf = xgb.XGBRegressor(**params)
            
        # fit the model
        clf.fit(X_train, y_train, 
                eval_set=eval_set,
                early_stopping_rounds = params["early_stopping_rounds"],
                verbose= params["verbose"])
 
        # save model 
        pickle.dump(clf, open("/".join([args["path"], args["date"], "02 results", args["cover"], "python_models", "model_{0}.pkl".format(i)]), 'wb'))
        
        # save feature importance into dataset importance
        #dataset_importance["importance"] = clf.feature_importances_
      
        object_cols = [x for x in X_test.columns if X_test[x].dtype == "O"]
        
        if params["objective"] == "binary:logistic" or params["objective"] == "multi:softmax":
            preds = clf.predict_proba(X_test.drop(['sample_number'] + object_cols, axis=1))
            pp = pd.DataFrame(np.transpose([y_test.tolist(), preds[:,1].tolist()]), columns = ["true", "pred"])
            new_residual = pd.concat([X_test.reset_index(drop=True), pp], axis=1)
            new_residual["Fold"] = i
            residuals = pd.concat([residuals, new_residual], axis = 0).reset_index(drop=True)
            
            fpr, tpr, thresholds = metrics.roc_curve(y_test, preds[:,1], pos_label=1)
            s = np.where(pp["true"] == 1, 1,-1)
            deviance = (s*np.sqrt(-2*(pp["true"]*np.log(pp["pred"]) + (1 - pp["true"])*np.log(1-pp["pred"])))).sum()
            print("[Fold {0}] AUC : {1}, logloss : {2}, Deviance {3}".format(i , metrics.auc(fpr, tpr), metrics.log_loss(y_test, preds), deviance))
            _folds.loc["Fold_%i"%(i+1), "AUC"] =  metrics.auc(fpr, tpr)
            _folds.loc["Fold_%i"%(i+1), "logloss"] = metrics.log_loss(y_test, preds)
            _folds.loc["Fold_%i"%(i+1), "Deviance"] = deviance
        
        elif params["objective"] == "reg:linear":
             preds = clf.predict(X_test.drop(['sample_number']+ object_cols, axis=1))
             if  max(data[Y_label])> 100:
                pp = pd.DataFrame(np.transpose([np.exp(y_test).tolist(), np.exp(preds).tolist()]), columns = ["true", "pred"])
             else:
                pp = pd.DataFrame(np.transpose([y_test.tolist(), preds.tolist()]), columns = ["true", "pred"])
            
             new_residual = pd.concat([X_test.reset_index(drop=True), pp], axis=1)
             new_residual["Fold"] = i
             residuals = pd.concat([residuals, new_residual], axis = 0).reset_index(drop=True)
             print("[Fold {0}] MAPE : {1}, RMSE : {2}, MAE {3}".format(i ,(abs(pp["true"] - pp["pred"])*100/pp["true"]).mean(), np.sqrt(mean_squared_error(pp["true"], pp["pred"]) ), abs(pp["true"] - pp["pred"]).mean()  ))
        
             _folds.loc["Fold_%i"%(i+1), "MAPE (%)"] = round((abs(pp["true"] - pp["pred"])*100/pp["true"]).mean(),2)
             _folds.loc["Fold_%i"%(i+1), "RMSE"] = round(np.sqrt(mean_squared_error(pp["true"], pp["pred"]) ),2)
             _folds.loc["Fold_%i"%(i+1), "MAE (AU$)"] = round(abs(pp["true"] - pp["pred"]).mean(),2)
             
        else:
            raise Exception("Wrong objective function")
            
    folds_list = ["Fold_{0}".format(x) for x in range(params["k_fold"])]
    if params["objective"] == "binary:logistic" or params["objective"] == "multi:softmax":
        _folds.loc["Total", "AUC"] = _folds.loc[folds_list,"AUC"].mean()
        _folds.loc["Total", "logloss"] = _folds.loc[folds_list,"logloss"].mean()
        _folds.loc["Total", "Deviance"]  = _folds.loc[folds_list,"Deviance"].mean()
    elif params["objective"] == "reg:linear":
        _folds.loc["Total", "MAPE (%)"] = _folds.loc[folds_list,"MAPE (%)"].mean()
        _folds.loc["Total", "RMSE"] = _folds.loc[folds_list,"RMSE"].mean()
        _folds.loc["Total", "MAE (AU$)"]  = _folds.loc[folds_list,"MAE (AU$)"].mean()
    else:
            raise Exception("Wrong objective function")
    
    #### save importance into results
    plot_importance_to_file(args, X_train, clf)
    
    #### save decile error
    if params["objective"] == "binary:logistic" or params["objective"] == "multi:softmax":
        s = np.where(residuals["true"] == 1, 1,-1)
        residuals["residuals"] = s*np.sqrt(-2*(residuals["true"]*np.log(residuals["pred"]) + (1 - residuals["true"])*np.log(1-residuals["pred"])))
    else:
         residuals["residuals"] = abs(residuals["pred"] - residuals["true"])*100/residuals["true"]
         
    print("_"*40)
    if params["objective"] == "reg:linear":
        print("[OVERALL] MAPE : {0:.3f}, RMSE : {1:.3f}, MAE {2:.3f} \n".format(_folds.loc["Total", "MAPE (%)"], _folds.loc["Total", "RMSE"], _folds.loc["Total", "MAE (AU$)"]))
    elif params["objective"] == "binary:logistic" or params["objective"] == "multi:softmax":
        print("[OVERALL] AUC : {0:.4f}, logloss : {1:.4f}, Deviance {2:.4f} \n".format(_folds.loc["Total", "AUC"], _folds.loc["Total", "logloss"], _folds.loc["Total", "Deviance"]))
    else:
            raise Exception("Wrong objective function")
            
    residuals.index = index

    return clf, residuals.sort_index(), dataset_importance.sort_values("importance"), _folds


def plot_importance_to_file(args, X, clf):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (11, max(11, int(X.shape[1]/3))))
#    explainer = shap.TreeExplainer(clf)
#    shap_values = explainer.shap_values(X)
#    shap.summary_plot(shap_values, X)
    xgb.plot_importance(clf, ax=ax)
    fig.savefig("/".join([args["path"], args["date"], "02 results", args["cover"], "images", "feature_importance.png"]), bbox_inches='tight', pad_inches=0)
    
    
# =============================================================================
# Grid search
# =============================================================================
def train_grid_search(parameters, key, X, y):
    
    params = parameters
    
    liste_parameters = list(key.keys())
    liste_parameters_value =  list(key.values())
    
    if len(liste_parameters_value) >1:
        combinations = list(itertools.product(*key.values()))
    else:
        combinations = liste_parameters_value[0]
        
    if parameters["verbose"] ==1:
        print("--> {0} fits for {1} fine tuning".format(len(combinations), tuple(liste_parameters)))
        
    for p, paire in enumerate(combinations):
        kpi = []
        
        if len(liste_parameters_value) >1:
            for i in range(len(paire)):
                params[liste_parameters[i]] = paire[i]
        else:
             params[liste_parameters[0]] = paire

        for i, value in k_fold_split(X, 4).items():
            X_train, X_test = X.loc[value["Train"]], X.loc[value["Test"]]
            y_train, y_test = y.loc[value["Train"]], y.loc[value["Test"]]
    
            eval_set  = [(X_train,y_train), (X_test,y_test)]
            if params["objective"] == "binary:logistic" or params["objective"] == "multi:softmax":
                clf = xgb.XGBClassifier(**params)
            else:
                clf = xgb.XGBRegressor(**params)
                
            clf.fit(X_train, y_train, 
                    eval_set=eval_set,
                    early_stopping_rounds = params["early_stopping_rounds"],
                    verbose= 0)
            
            kpi.append(clf.best_score)
            
        avg_mae = np.mean(kpi)
        if parameters["verbose"] ==1:
                print("[{0}/{1}] Grid Search {2} = {3} : {5} = {4:.4f}".format(p+1, len(combinations), tuple(liste_parameters), paire, avg_mae, parameters["eval_metric"]))    
        
        if p ==0:   
            ref_mae = avg_mae
            best_params = paire
            
        if avg_mae < ref_mae:
            ref_mae = avg_mae
            best_params = paire
    
    results = {}
    
    if parameters["verbose"] ==1:
        print("Best selected param : {0} : {1} with {3} :  {2}".format(tuple(liste_parameters), best_params, ref_mae, parameters["eval_metric"]))
    
    if len(liste_parameters_value) >1:
        for i in range(len(liste_parameters)):
            results[liste_parameters[i]] = best_params[i]
    else:
        results[liste_parameters[0]] = best_params
    
    return results


# =============================================================================
# FINE TUNING PARAMETERS
# =============================================================================
def finetuning(args, data, Y_label, parameters):
    
    print("\n" + "--"*10 + str("  Grid search of xgb parameters  ") + "--"*10 +" \n")
    parameters["verbose"] = 1     
    t0  = time.time()
    data = data.reset_index(drop=True).copy()
    grid = [{"learning_rate" : list(np.arange(parameters["learning_rate"]*0.5, parameters["learning_rate"]*2, parameters["learning_rate"]*0.1))},
             {"max_depth" : range(parameters["max_depth"] -2 , min(8, parameters["max_depth"]  + 2)),
              "min_child_weight": range(max(5,parameters["min_child_weight"] - 4), parameters["min_child_weight"] + 4, 2)},
             {"colsample_bytree" : list(np.linspace(max(0.1,parameters["colsample_bytree"]-0.10), min(1,parameters["colsample_bytree"]+0.15), num=5)),
                 "subsample": list(np.linspace(max(0.1,parameters["subsample"]-0.10), min(1,parameters["subsample"]+0.10), num=4))},
              {"gamma" : range(max(0, parameters["gamma"]-2),parameters["gamma"]+4)}]

    for key in grid:
        if max(data[Y_label])> 100:
            y = np.log(data[Y_label])
        else:
            y = data[Y_label]
        results = train_grid_search(parameters, key, data.drop(["sample_number", Y_label], axis= 1), y)
        for key, value in results.items():
            parameters[key] = value
            
    parameters["verbose"] = 0       
    print("Final parameters set is : {0} in time : {1}".format(parameters, time.time() - t0))
    
    return parameters


# =============================================================================
# Vehicle grouping/clustering
# =============================================================================
def create_clusters(df, keys, score, resp, ncluster=20, w=None, type='kmenoids', tolerance=0.001):
    from pyclustering.cluster.kmedoids import kmedoids
    from sklearn.cluster import AgglomerativeClustering
    
    grouped = df.groupby(keys)[score].mean()
    
    if w is not None:
        grouped_w = df.groupby(keys)[w].sum()
        df_vs=keys + [resp, w]
    else:
        w='count'
        grouped_w = df.groupby(keys)[score].count()
        grouped_w.name=w
        df_vs = keys + [resp]
        
    if type=='kmenoids':
        calculate_init = pd.concat([grouped, grouped_w], axis=1)
        calculate_init['index'] = list(range(len(grouped)))
        calculate_init = calculate_init.sort_values(by=score)
        calculate_init['cw'] = calculate_init[w].cumsum().div(calculate_init[w].sum())
        quantiles = np.linspace(0, 1, ncluster + 2)[1:-1]
        init_centroid = list(map(lambda x: calculate_init[calculate_init['cw'] > x]['index'].iloc[0], quantiles))
        clustering = kmedoids(grouped.values.reshape(-1, 1).tolist(), init_centroid,
                                                            tolerance=tolerance)
        clustering.process()
        clusters = clustering.get_clusters()
        cluster_mapping = {index: n for n, instance in enumerate(clusters) for index in instance}

    elif type=='ward':
        ff = np.average #lambda x: np.average(x, w=df[w].iloc[x.index])
        clusters= AgglomerativeClustering(n_clusters=ncluster,pooling_func=ff)
        cluster_values=clusters.fit_predict(grouped.values.reshape(-1, 1))
        cluster_mapping=dict(zip(range(0,len(grouped)),cluster_values))
        clusters=range(0,ncluster)
        
    grouped = grouped.to_frame().reset_index()
    grouped['cluster'] = grouped.index.map(lambda x: cluster_mapping.get(x, None))
    merged = pd.merge(df[df_vs], grouped, left_on=keys, right_on=keys, how='left').reset_index(False)
    reoder_cluster = {i: n for n, i in enumerate(merged.groupby('cluster')[resp].aggregate(
        lambda x: np.average(x, weights=merged.loc[x.index,w])).sort_values().index)}
    return merged['cluster'].map(reoder_cluster).values    



def clustering(args, data, Y_label= "RESPONSE", params= {}):
    
    if np.max(data[args["glm_random_split"]]) == 100:
        cap = 80
    elif np.max(data[args["glm_random_split"]]) == 1:
        cap = 0.8
    elif np.max(data[args["glm_random_split"]]) == 10:    
        cap = 8
    else:
        raise Exception(" Please create the random variable such that values are between 0-100, 0-10 or 0-1")
        
    if params["objective"] == "binary:logistic":
        data["GLM_PREDICTION"] = np.log(data["GLM_PREDICTION"]) - np.log(1- data["GLM_PREDICTION"])
    else:
        data["GLM_PREDICTION"] = np.log(data["GLM_PREDICTION"])
        
    train = xgb.DMatrix(data.loc[data[args["glm_random_split"]] <=cap].drop([Y_label, args["glm_random_split"], "GLM_PREDICTION", "veh_key"], axis =1), data.loc[data[args["glm_random_split"]] <=cap][Y_label])
    test = xgb.DMatrix(data.loc[data[args["glm_random_split"]] >cap].drop([Y_label, args["glm_random_split"], "GLM_PREDICTION", "veh_key"], axis =1), data.loc[data[args["glm_random_split"]] >cap][Y_label])
    train.set_base_margin(data.loc[data[args["glm_random_split"]] <=cap]["GLM_PREDICTION"])
    test.set_base_margin(data.loc[data[args["glm_random_split"]] >cap]["GLM_PREDICTION"])
    
    watchlist = [(train, 'train'), (test, 'eval')]
    
    clf = xgb.train(params, train, evals= watchlist, 
                    verbose_eval= params["verbose_eval"], 
                    num_boost_round = params["n_estimators"], 
                    early_stopping_rounds = params["early_stopping_rounds"])
    y_pred = clf.predict(test)
    
    # plot roc curves GLM vs GBM for test set
    auc_gbm, auc_glm = auc_roc_curves_gbm_glm(args, data.loc[data[args["glm_random_split"]] >cap], y_pred, Y_label)
    
    # plot feature importance
    plot_importance_to_file(args, clf, data.shape[1])
    
    importance = pd.DataFrame(np.transpose([list(clf.get_fscore().keys()), list(clf.get_fscore().values())]), columns = ["features", "f2_score"])
    importance["f2_score"] =  importance["f2_score"].astype(int)
    importance = importance.sort_values(by = "f2_score",ascending = 0)
    
    #### predict the margin of the model. It gives straight away the pure vehicle effect if no vehicle effect come from the GLM  
    p = pd.DataFrame(clf.predict(xgb.DMatrix(data.drop([Y_label, args["glm_random_split"], "GLM_PREDICTION", "veh_key"], axis =1)), output_margin = True))
    data["GBM_PREDICTION"] = p[0]
    
    #### check variance per veh_key is low  as groupby will be done per veh_key
    agg = data[["veh_key", "GBM_PREDICTION"]].groupby('veh_key').std()
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    agg['GBM_PREDICTION'].hist(bins=50, figsize=(10,10))
    fig.savefig("/".join([args["path"], args["date"], "02 results", args["cover"], "images", "standar_deviation_veh_key.png"]), bbox_inches='tight', pad_inches=0)
    print(agg['GBM_PREDICTION'].mean())
    
    vehicle_table= data.groupby('veh_key').aggregate({'GBM_PREDICTION': 'mean',
                                                      Y_label:'sum',
                                                      args["glm_random_split"] : 'count'}).reset_index()
    
    vehicle_table[Y_label] = vehicle_table[Y_label] / vehicle_table[args["glm_random_split"]]
    vehicle_table["weight"] = 1
    
    n_cluster = 20
    vehicle_table['cluster_XGB']= create_clusters(df=vehicle_table,
                                                     keys=['veh_key'], 
                                                     score='GBM_PREDICTION', 
                                                     resp=Y_label, 
                                                     ncluster=n_cluster, 
                                                     w= "weight",
                                                     type='kmenoids', 
                                                     tolerance=0.05)
    
    fig, ax1 = plt.subplots(figsize = (10,10))
    ax = vehicle_table[['cluster_XGB', args["glm_random_split"]]].groupby('cluster_XGB').sum().plot(kind='bar', stacked=True,  legend=False, alpha = 0.6, color= 'y', ax = ax1)
    vehicle_table[['cluster_XGB', 'RESPONSE']].groupby('cluster_XGB').mean().plot(secondary_y=True, ax=ax, alpha = 0.85, style='bo--')
    ax.autoscale(tight=False) 
    fig.savefig("/".join([args["path"], args["date"], "02 results", args["cover"], "images", "cluster_output_average.png"]), bbox_inches='tight', pad_inches=0)
    
    data["pred"] = p[0]
    
    _folds = pd.DataFrame([], index = ["Test set"], columns = ["AUC_GBM", "AUC_GLM", "Shape"])
    _folds.loc["Test set"]  = [auc_gbm, auc_glm, len(y_pred)]
    
    return clf, vehicle_table, importance, _folds


""" tea teat aat at ae t"""