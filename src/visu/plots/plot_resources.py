# -*- coding: utf-8 -*-
"""
Created on Mon Oct 09 09:50:11 2017

@author: alexandre
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import operator
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
from xgboost import plot_importance

"""
Done:
    - variables vs Y_label with bandings available and average per variable
    - Importance variable plots
    - AUC plot
"""

"""
To do:
    - Plot correlations
    - Plot PCA importance variables ---> eigen values
"""

def var_vs_target(data, Y_label, variable, bins=30):
    
    if type(Y_label) == str:
        Y_label = [Y_label]
    data = data.copy()
    
    if variable not in data.columns:
        return "variable not in database"
   
    if len(data[variable].value_counts())>bins:
        if data[variable].dtype !="O":
            data[variable] = pd.cut(data[variable] , bins, precision = 1)
        else:
            modalities = data[variable].value_counts().index[:bins]
            data.loc[~data[variable].isin(modalities), variable] = "other"
        
    avg_target = data[Y_label].mean()
    Y = data[[variable] + list(Y_label)].groupby(variable).mean()
    P = data[[variable] + list(Y_label)].groupby(variable).agg([np.size, np.std])
    
    ### add confidence_interval
    plt.figure(figsize= (12,8))
    
    ax1 = P[Y_label[0]]["size"].plot(kind="bar", alpha= 0.42)
    ax2 = ax1.twinx()
    s = ax2.plot(ax1.get_xticks(), Y[Y_label], linestyle='-', label= [Y_label])
    
    ax1.set_ylabel('%s Volume'%str(variable))
    ax2.set_ylabel('%s'%str(Y_label))
    ax1.set_xlabel('%s'%str(variable))
    
    plt.title("Evolution of %s vs %s"%(variable, Y_label))
    ax2.legend(tuple(Y_label), loc= 1, borderaxespad=0.)
    
    for i, value in enumerate(avg_target):
        plt.axhline(y=value, xmin=0, xmax=3, linewidth=0.5, linestyle="--", color = s[i].get_color())
        plt.errorbar(ax1.get_xticks(), Y[Y_label[i]], yerr=1.96*P[Y_label[i]]["std"]/np.sqrt(P[Y_label[0]]["size"]), alpha= 0.65, color= s[i].get_color())
    plt.setp(ax1.xaxis.get_ticklabels(), rotation=78)
    plt.show()
    
    return {"size/std" : P, "mean" : Y}


#### plot variable importance, mainly for tree based algorithms
def variable_importance_plots(clf):

    ax = plot_importance(clf, height=0.5)
    fig = ax.figure
    fig.set_size_inches((8,14))
    plt.show()
    

### plot AUC curve based on a pretrained model clf
def auc_plot(data, Y_label, clf):
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, threshold = roc_curve(data[Y_label]+1, clf.predict_proba(data.drop(Y_label, axis=1))[:,1], pos_label = 2 )
    auc_ = auc(fpr,tpr)
    
    ##### plot auc roc curve 
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.5f)' % auc_)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    return auc_


def concurrent_variables(data, X, Ys, category = None):  
    
    if type(Ys) == str:
        Ys = [Ys]
        
    plt.figure(figsize=(10,10))
    
    if not category:
        for i, Y in enumerate(Ys):
            plt.scatter(data[X], data[Y], s=10)
        plt.legend(tuple(Ys), loc= 1, borderaxespad=0., fontsize='small')
    
    else:
        groups = data[[X] + Ys + [category]].groupby(category)

        # Plot
        fig, ax = plt.subplots()
        fig.figsize = (10,10)
        ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
        for name, group in groups:
            for i, Y in enumerate(Ys):
                ax.plot(group[X], group[Y], marker='o', linestyle='', ms=2, label=name)
        ax.legend(loc= 1, borderaxespad=0., fontsize='small')
        
    plt.xlabel('%s'%X)
    plt.ylabel('%s'%Ys)
    plt.title(' Concurrent variables: %s vs %s'%(X, Ys))
    plt.plot([0, np.max(data[X])], [0, np.max(data[X])], linewidth=0.5, linestyle="--")
    plt.show()
    
    
def data_on_map(data, Y_label, country= "australia", bar_max= None):
    
    """
    urcrnrlat=y1, llcrnrlat=y2, llcrnrlon=x1, urcrnrlon=x2
    e.g australia : (y1,y2,x1,x2) = (urcrnrlat=-9, llcrnrlat=-41, llcrnrlon=105, urcrnrlon=153)
    """
    
    if country == "australia":
        (y1,y2,x1,x2) = (-9, -41, 105, 153)
    
    plt.figure(figsize = (13,13))
    m = Basemap(urcrnrlat=y1, llcrnrlat=y2, llcrnrlon=x1, urcrnrlon=x2,
       projection='tmerc', resolution = "i", lat_0 =(y1+y2)/2.0, lon_0 = (x1+x2)/2.0)
    
    x,y = m(data["lon"].tolist(), data["lat"].tolist())
    
    if bar_max:
        m.scatter(x,y, marker = "o", s=10, c = data[Y_label], cmap = cm.cool, vmin=0, vmax= bar_max)
    else:
        m.scatter(x,y, marker = "o", s=10, c = data[Y_label], cmap = cm.cool)
        
    # draw the borders of the map
    m.drawmapboundary()
    # draw the coasts borders and fill the continents
    m.drawcoastlines()
    m.drawstates(color = "0.6")
    plt.title('Average %s per postcode'%str(Y_label))
    plt.colorbar(shrink = 0.60)
    
#    if country == "australia": 
#        plt.annotate("NSW", xy=(146.92,31.25), xytext = (146.92, 31.25), xycoords= "data")
    
    plt.show()
    
