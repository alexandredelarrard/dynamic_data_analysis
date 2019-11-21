# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:19:30 2019

@author: JARD
"""

import seaborn as sns; sns.set()
from matplotlib import pyplot as plt
import pandas as pd

try:
    from pandas.tools.plotting import table
except ModuleNotFoundError:
    from pandas.plotting import table
    
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import re


plt.rcParams["figure.figsize"] = (10,12)

# =============================================================================
# General plot functions for dataprep understanding
# =============================================================================
def explore_output_rate(data, var, Y_label):
    '''
    This function plot the average of the output depending on each modality of the variable setted in parameter var
        Inputs: data = dataset name
                var = Name of variable to study, as a string
                Y_LABEL = Name of the output, as a string
    '''
    
    data = data.copy()
    if data[var].dtype != "O": #### if the type of variable is object (string elements) then no need to band them into percentiles of 2%
        data[var + "_cuts"] = pd.qcut(data[var], q = 50, duplicates= "drop") ### band variable var into 50 groupes of 2 % and drop duplicated groupes (those having same banding rules)
    else:
        data[var + "_cuts"] = data[var]
        
    a = data[[Y_label, var + "_cuts"]].groupby(var + "_cuts").mean() ### equivalent to sas sql groupby : we get the average of var + "_cuts" variable (eg: TOT_SUM_INS_cuts) per modality
    volume = data[var + "_cuts"].value_counts() #### get the volume of each modality for variable var + "_cuts"
    a["volume"] = volume
    a[Y_label].plot(rot = 75, figsize = (11,7), title = var, secondary_y=True) ### pandas plot. a is here a dataframe
    a["volume"].plot(kind="bar", color = "gold")
    plt.show()
    

def plot_correlations(numerical_data):
     ### plot correlations
    fig, ax = plt.subplots(1,1, figsize=(15,12))
    corr= numerical_data.corr()
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True,  vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax = ax)    
    
    return corr


def var_vs_target(data, Y_label, variable, bins=30, normalize = False):
    
    if type(Y_label) == str:
        Y_label = [Y_label]
        
    data = data.copy()
    
    if variable not in data.columns:
        return "variable not in database"
   
    if len(data[variable].value_counts())>bins:
        if data[variable].dtype !="O":
            data[variable] = pd.qcut(data[variable] , bins, precision = 1, duplicates = "drop")
        else:
            modalities = data[variable].value_counts().index[:bins]
            data.loc[~data[variable].isin(modalities), variable] = "other"
        
    avg_target = data[Y_label].mean()
    if normalize:
        Y = data[[variable] + list(Y_label)].groupby(variable).mean() / data[list(Y_label)].mean()
        
    else:
        Y = data[[variable] + list(Y_label)].groupby(variable).mean()
        
    P = data[[variable] + list(Y_label)].groupby(variable).agg([np.size, np.std])
    
    ### add confidence_interval
    plt.figure(figsize= (12,8))
    
    ax1 = P[Y_label[0]]["size"].plot(kind="bar", alpha= 0.42, grid= True)
    ax2 = ax1.twinx()
    
    if normalize:
        ax2.set_ylim([np.min(np.min(Y))*0.95, np.max(np.max(Y))*1.05])
    
    s = ax2.plot(ax1.get_xticks(), Y[Y_label], linestyle='-', label= [Y_label])
    
    ax1.set_ylabel('%s Volume'%str(variable))
    ax2.set_ylabel('%s'%str(Y_label))
    ax1.set_xlabel('%s'%str(variable))
    
    plt.title("Evolution of %s vs %s"%(variable, Y_label))
    ax2.legend(tuple(Y_label), loc= 1, borderaxespad=0.)
    
    if not normalize:
        for i, value in enumerate(avg_target):
            plt.axhline(y=value, xmin=0, xmax=3, linewidth=0.5, linestyle="--", color = s[i].get_color())
            plt.errorbar(ax1.get_xticks(), Y[Y_label[i]], yerr=1.96*P[Y_label[i]]["std"]/np.sqrt(P[Y_label[0]]["size"]), alpha= 0.65, color= s[i].get_color())
    
    plt.setp(ax1.xaxis.get_ticklabels(), rotation=78)
    plt.show()
    
    return {"size/std" : P, "mean" : Y}

# =============================================================================
# REGRESSION ERRORS plots per classe
# =============================================================================
def true_vs_preds(preds, args):
    '''
    Plot a scatter plot between the true value and its prediction 
    '''
    fig, ax = plt.subplots()
    ax.scatter(preds["true"], preds["pred"], s= 5, alpha= 0.5)
    lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
    
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, color= "red")
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.ylabel("Predicted Premium")
    plt.xlabel("True Premium")
    fig.savefig("/".join([args["path"], args["date"], "02 results", args["cover"], "images", "true_pred.png"]), bbox_inches='tight', pad_inches=0)
    

def true_vs_preds_by_hue(preds, args, feature):
    
    #### same for multi purpose
    g = sns.lmplot(x="true", y="pred", hue=feature, data=preds, markers="o", size= 10, scatter_kws={"s": 25})
    if "max_premium" in args.keys():
        g.set(ylim= (0, args["max_premium"]))
        g.set(xlim=(0, args["max_premium"]))
    g.savefig("/".join([args["path"], args["date"], "02 results", args["cover"], "images", "true_pred_per_insurer.png"]), bbox_inches='tight', pad_inches=0)
    
    
def true_vs_preds_each_hue(preds, args, feature):
    
    for ins in preds[feature].unique():
        sub = preds.loc[preds[feature] == ins]
        g = sns.lmplot(x="true", y="pred", hue=feature, data=sub, markers="o", size= 10, scatter_kws={"s": 25})
        if "max_premium" in args.keys():
            g.set(ylim= (0, args["max_premium"]))
            g.set(xlim=(0, args["max_premium"]))
        g.savefig("/".join([args["path"], args["date"], "02 results", args["cover"], "images", "true_vs_pred_%s.png"%ins]), bbox_inches='tight', pad_inches=0)


def error_per_premium_decile(preds, args):
    
    if "Insurer" not in preds.columns:
         preds["Insurer"] = "Allianz"
    
    ### error per premium level percentage and aud
    preds["output"] = pd.cut(preds["true"], 30, right= True)
    gp = preds[["output", "pred", "true", "Insurer"]].groupby(["output", "Insurer"]).mean()
    gp["percentage_error"] = abs(gp["pred"] - gp["true"])*100/gp["true"]
    gp["percentage_error"].unstack(level=1).plot(subplots=False, rot= 45, title = "% error per output level")
    plt.ylabel("% error")
    plt.savefig("/".join([args["path"], args["date"], "02 results", args["cover"], "images", "error_percent_insurer.png"]), bbox_inches='tight', pad_inches=0)
        
    gp["aud_error"] = abs(gp["pred"] - gp["true"])
    gp["aud_error"].unstack(level=1).plot(subplots=False, rot= 45, title = "AUD error per output level")
    plt.ylabel("AUD error")
    plt.savefig("/".join([args["path"], args["date"], "02 results", args["cover"], "images", "error_dollar_insurer.png"]), bbox_inches='tight', pad_inches=0)


def mape_error_by_feature(preds, args, feature):
    
    if "Insurer" not in preds.columns:
        preds["Insurer"] = "Allianz"
    
    preds["absolute_error"] = abs(preds["true"] - preds["pred"])
    preds["MAPE(%)"] = abs(preds["true"] - preds["pred"])*100/preds["true"]
    agg = preds.groupby(["Insurer", feature]).mean()
    agg["count"] = preds[["Insurer", feature, "true"]].groupby(["Insurer", feature]).size()
    
    agg2 = preds.groupby(["Insurer"]).mean()
    agg2["volume_tested"] = preds.groupby(["Insurer"]).size()
   
    ### percentage error per insurer and state
    fig, ax = plt.subplots(1, 1)
    fig = agg["count"].unstack(level=0).plot(
        kind='bar',
        stacked=True,
        alpha= 0.4,
        ax=ax
    )
    
    table(ax, np.round(agg2[["true", "pred", "volume_tested", "MAPE(%)"]], 2), loc='top', colWidths=[0.15, 0.15, 0.15, 0.15])
    fig.set_ylabel("volume of tested premium", fontsize=17)
    fig.set_xlabel("state",fontsize=17)
    
    fig2 = agg["MAPE(%)"].unstack(level=0).plot(
        linewidth=2.0,
        mark_right=False,
        ax=ax,
        secondary_y=True, marker = "o"
    )
    fig2.set_ylabel("MAPE(%)", fontsize=17)
    plt.savefig("/".join([args["path"], args["date"], "02 results", args["cover"], "images", "error_percent_insurer_feature.png"]), bbox_inches='tight', pad_inches=0)
    
    ### aud error per insurer and state
    fig, ax = plt.subplots(1, 1)
    fig = agg["count"].unstack(level=0).plot(
        kind='bar',
        stacked=True,
        alpha= 0.4,
        ax=ax
    )
    
    table(ax, np.round(agg2[["true", "pred", "volume_tested", "absolute_error"]], 2), loc='top', colWidths=[0.15, 0.15, 0.15, 0.15])
    fig.set_ylabel("volume of tested premium", fontsize=17)
    fig.set_xlabel("state",fontsize=17)
    fig2 = agg["absolute_error"].unstack(level=0).plot(
        linewidth=2.0,
        mark_right=False,
        ax=ax,
        secondary_y=True, marker = "o"
    )
    fig2.set_ylabel("absolute average error in AU$", fontsize=17)
    plt.savefig("/".join([args["path"], args["date"], "02 results", args["cover"], "images", "error_dollar_insurer_feature.png"]), bbox_inches='tight', pad_inches=0)
    
    return agg, preds


# =============================================================================
# partial dependency plots
# =============================================================================

def pdp_one_variable(clf, X, feature_name):
    """
    Calculate the dependency (or partial dependency) of a response variable on a predictor (or multiple predictors)
    1. Sample a grid of values of a predictor.
    2. For each value, replace every row of that predictor with this value, calculate the average prediction.
    """
    
    grid = np.sort(X[feature_name].unique())
    
    if len(grid) >30:
        intervals = list(pd.qcut(grid, 30, duplicates="drop").value_counts().index)
        grid = [np.round((x.left+ x.right)/2, 1) for x in intervals]
        
    grid.sort()
    print("variable : {0}, number modalities {1}".format(feature_name, len(grid)))
    insurer_variables = [x for x in X.columns if "Insurer_" in x]
    
    ### get pdp for each available insurer
    if len(insurer_variables) > 0:
        pdp = pd.DataFrame([], columns = insurer_variables, index = grid)
        
        for ins in insurer_variables:
            X_temp = X.copy()
            
            ### 1 for the insurer studied, 0 otherwise
            X_temp[ins]  = 1
            X_temp[[x for x in insurer_variables if x != ins]] = 0
            
            for i, val in enumerate(grid):
                    X_temp[feature_name] = val
                    if clf.objective == 'binary:logistic':
                        preds= clf.predict_proba(X_temp)[:,1]
                    else:
                        preds = clf.predict(X_temp)
                    pdp.loc[val, ins] = np.mean(preds)
            
    else:
        pdp = pd.DataFrame([], columns = ["Allianz"], index = grid)
        X_temp = X.copy()
        for i, val in enumerate(grid):
                X_temp[feature_name] = val
                if clf.objective == 'binary:logistic':
                    preds= clf.predict_proba(X_temp)[:,1]
                else:
                    preds = clf.predict(X_temp)
                pdp.loc[val, "Allianz"] = np.mean(preds)
                
    return pdp
    

def partial_dependency(clf, X, dico_prep):
    
    output = {}
    feature_name = [x for x in clf.get_booster().feature_names]
    X_temp = X[feature_name].copy()
    dico_prep.index = [x.lower() for x in dico_prep.index]

    for feature in tqdm(feature_name):
        if "Insurer_" not in feature:
            try:
                output[feature.lower()] = pdp_one_variable(clf, X_temp, feature)
            except Exception as e:
                print("Following error/missing variable for pdp: {} ".format(e))
        
    ### transform dummy variables back to their original modality names
    for f in feature_name:
        f = f.lower()
        if f in dico_prep.index:
            output[f].index = [dico_prep.loc[f][output[f].index[0]], dico_prep.loc[f][output[f].index[1]]]
          
    ### transform back dummified variables into one variable with all modalities. 
    ### It enables to have one pdp for categorical variables
    col_dumy = dico_prep["dummyfied"].apply(lambda x: len(x) > 1)
    for col in col_dumy.loc[col_dumy].index:
        new_pdp = pd.DataFrame([])
        for modality in dico_prep.loc[col, "dummyfied"].tolist():
            modality = re.sub('[ -]','_', str(modality))
            if modality != "nan" and not pd.isnull(modality) and col+ "_" + modality in output.keys():
                try:
                    pdp = output[col+ "_" + modality]
                    for ins in pdp.columns:
                        new_pdp.loc[modality, ins] = output[col+ "_" +  modality].loc[1,ins]
                    output.pop(col+ "_" +  modality, None)  
                
                except Exception as e:
                    print(e)
                    pass
                
        output[col] = new_pdp
        
    return output


def pdp_plots_by_feature(args, template_vars, dico_prep, preds, clf):
    '''
    Plot all pdp that are stored into my_plots along with its volume. my_plots is a dictionnary of dataframes
    each dataframe is linked to one feature, each observation in this dataframe is a modality linked to the averaged output of the dataset when all observation have the modality as description factor
    all plots are saved into "/".join([args["path"], args["date"], "02 results", args["cover"], "images", "pdp_%s.png"%f])
    '''
    print(" Create partial dependency plots ")
    preds = preds.copy()
    my_plots = partial_dependency(clf, preds, dico_prep)
    template_vars["plot_features"] = []
    preds.columns= [x.lower() for x in preds.columns] 
    
    for i, f in enumerate(my_plots.keys()):
        if not my_plots[f].empty:
            volumes= []
            if f in dico_prep.index:
                if len(dico_prep.loc[f, "dummyfied"])>1:
                    for moda in my_plots[f].index:
                        volumes.append(sum(preds[f+"_" + moda.lower().replace(" ","_")]))
                else:
                    volumes= preds[f].value_counts().sort_index().values
            else:
                modalities =  my_plots[f].index.tolist()
                if len(modalities) >=30: 
                    volumes.append(sum(preds[f].between(preds[f].min(), (modalities[0]+ modalities[1])/2)))
                    for p in range(1, len(modalities) - 1):
                        volumes.append(sum(preds[f].between((modalities[p-1]+modalities[p])/2, (modalities[p]+modalities[p+1])/2)))
                    volumes.append(sum(preds[f].between((modalities[-2]+modalities[-1])/2, preds[f].max())))
                else:
                    volumes= preds[f].value_counts().sort_index().values
                
            try:
                fig, ax1 = plt.subplots()
                
                ax1.plot(range(len(my_plots[f])), my_plots[f].values, 'o-')
                ax1.set_xticks(range(len(my_plots[f])))
                ax1.set_xticklabels(my_plots[f].index, rotation=60)
                ax1.margins(0.03)
                ax1.set_title(f.upper())
                ax1.set_ylabel('Average Output')
                ax1.legend(my_plots[f].columns)
                
                ax2 = ax1.twinx()
                ax2.bar(ax1.get_xticks(), volumes, alpha= 0.3, color = "gold")
                ax2.grid(False)
                
                if args["cover"] == "normalised_premium":
                    ax1.set_ylim([0.8,1.6])
                    
                plt.savefig("/".join([args["path"], args["date"], "02 results", args["cover"], "images", "pdp_%s.png"%f]), bbox_inches='tight', pad_inches=0)
                template_vars["plot_features"].append(args["path_results"].replace(" ","%20") + "/pdp_%s.png"%f)
            
            except Exception as e:
                print("error for {0}, my plot {2} / {1}".format(f, my_plots[f], e))
            
    return template_vars

# =============================================================================
# CLASSIFICATION ERROR plots and charts
# =============================================================================
def auc_per_feature(args, preds, feature = "Insurer"):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    if feature in preds.columns:
        for ins in preds[feature].unique():
            sub = preds.loc[preds[feature] == ins]
            fpr[ins], tpr[ins], _ = roc_curve(sub["true"], sub["pred"])
            roc_auc[ins] = auc(fpr[ins], tpr[ins])
        
    fpr["total"], tpr["total"], _ = roc_curve(preds["true"], preds["pred"])
    roc_auc["total"] = auc(fpr["total"], tpr["total"])
    
    plt.figure(figsize = (8,8))
    for key in fpr.keys():
        plt.plot(fpr[key], tpr[key],
                 label='{1} : AUC under ROC curve (area = {0:0.3f})'
                       ''.format(roc_auc[key], key), linewidth=2, alpha = 0.7)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if feature in preds.columns:
        plt.title('Area under the ROC curve per {0}'.format(feature))
    else:
        plt.title('Area under the ROC curve')
    plt.legend(loc="lower right")
    plt.savefig("/".join([args["path"], args["date"], "02 results", args["cover"], "images", "error_percent_insurer.png"]))


def auc_roc_curves_gbm_glm(args, X_preds, y_test, Y_label):
    
    #### evaluate error test set GLM vs GBM
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fpr = {}
    tpr = {}
    auc = {}
    fpr["GBM"], tpr["GBM"], thresholds = metrics.roc_curve(X_preds[Y_label], y_test, pos_label=1)
    auc["GBM"]=  metrics.auc(fpr["GBM"], tpr["GBM"])
    fpr["GLM"], tpr["GLM"], thresholds = metrics.roc_curve(X_preds[Y_label], X_preds["GLM_PREDICTION"], pos_label=1)
    auc["GLM"] =  metrics.auc(fpr["GLM"], tpr["GLM"])

    plt.plot(fpr["GBM"], tpr["GBM"], label='GBM : AUC under ROC curve (area = {0:0.3f})'.format(auc["GBM"]), linewidth=2, alpha = 0.7)
    plt.plot(fpr["GLM"], tpr["GLM"], label='GLM : AUC under ROC curve (area = {0:0.3f})'.format(auc["GLM"]), linewidth=2, alpha = 0.7)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Area under the ROC curve Train')
    plt.legend(loc="lower right")

    print("Test shape : {0}, AUC GLM {1} / AUC GBM {2}".format(y_test.shape[0], auc["GBM"], auc["GLM"]))
    plt.savefig("/".join([args["path"], args["date"], "02 results", args["cover"], "images", "error_percent_insurer.png"]))
    
    return auc["GBM"], auc["GLM"]
    
    
def confusion_matrix_plot(args, preds):
    
    ### take the threshold such that: if 10% of 1, we get 10% of predicted ones when higher than the thrershold 
    confus = confusion_matrix(preds["true"], (preds["pred"] >= np.percentile(preds["pred"], np.round(100 - 100*preds["true"].mean(), 1)))*1)
    df_cm = pd.DataFrame(confus, index = ["True Neg", "True Pos"],
                  columns = ["Predicted Neg", "Predicted Pos"])
    cmap = plt.get_cmap('Blues')
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.heatmap(df_cm, annot=True, cmap = cmap, square=True, linewidths=.5, cbar=False, fmt='g')
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
    ax.set_title("CONFUSION MATRIX : \n \nPrecision: {0:.2f}% \n Recall: {1:.2f}%".format(df_cm.loc["True Pos", "Predicted Pos"]*100/(df_cm.loc["True Pos", "Predicted Pos"] + df_cm.loc["True Neg", "Predicted Pos"]),
                                                                             df_cm.loc["True Pos", "Predicted Pos"]*100/(df_cm.loc["True Pos", "Predicted Pos"] + df_cm.loc["True Pos", "Predicted Neg"])))
    
    plt.savefig("/".join([args["path"], args["date"], "02 results", args["cover"], "images", "error_dollar_insurer.png"]))
    
    
def true_vs_pred_binary(args, preds):
    
    preds["pred_percentile"] = pd.cut(preds["pred"], 100,  precision = 1)
    preds["pred_percentile"] = preds["pred_percentile"].apply(lambda x : np.round((x.left + x.right)/2, 2))
    agg = preds[["true", "pred_percentile"]].groupby("pred_percentile").mean()
    vol = preds[["true", "pred_percentile"]].groupby("pred_percentile").size()
    
    fig, ax = plt.subplots()
    ax.plot(agg.index, agg.values, "b-o")
    ax.set_ylabel("True output")
    ax.set_xlabel("Predicted output")
    ax.set_title("True vs predictions per percentile")
    
    ax.plot([0,1], [0,1],"g--")
    
    ax2 = ax.twinx()
    vol.plot(alpha= 0.5, color = "gold", lw = 2)
    ax2.legend(["Predicted Probability distribution"], loc='upper left')
    ax2.grid(False)
    
    plt.savefig("/".join([args["path"], args["date"], "02 results", args["cover"], "images", "true_pred.png"]))
    