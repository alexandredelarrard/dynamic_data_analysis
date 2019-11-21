# -*- coding: utf-8 -*-
"""
Created on Thu May 17 12:27:45 2018

@author: JARD
"""

import numpy as np
import os
from matplotlib import pyplot as plt
import glob 
import pickle

try:
    from pip._internal import main
except Exception:
    from pip import main
plt.rcParams["figure.figsize"] = (11,11)

try:
    import xgboost as xgb
except ImportError:
    main(['install', r"N:\Computer\Python\xgboost-0.71-cp36-cp36m-win_amd64.whl"])
    import xgboost as xgb

try:
    import seaborn as sns; sns.set(color_codes=True)
except ImportError:
    main(['install', r"N:\Computer\Python\seaborn-0.8.1.tar.gz"])
    
try:
    from sklearn2pmml import sklearn2pmml
    from sklearn2pmml.pipeline import PMMLPipeline
except ImportError:
    main(['install', r"N:\Computer\Python\sklearn_pandas-1.6.0-py2.py3-none-any.whl"])
    main(['install', r"N:\Computer\Python\sklearn2pmml-master.zip"])
    main(['install', r"N:\Computer\Python\PyQt5-5.11.2-5.11.1-cp35.cp36.cp37.cp38-none-win_amd64.whl"])
    os.environ["PATH"] += ";N:\Computer\Java\jre1.8.0_171\bin"
 

def to_pmml_xgb(args, data, preds, Y_label, params):
    '''
    Create a pmml trained on 100% of the dataset. 
    --> if glm random split is not none then predict then whole dataset and add a column Train/test into the preds dataframe, 
        necessary to get the Test error and compare it with the GLM error
    '''
    
    if "Insurer" not in preds.columns:
        preds["Insurer"] = "Allianz"
    
    path_pmml = "/".join([args["path"], args["date"], "03 pmml", "comp_decomp_%s_xgb_%s.pmml"%(args["cover"], args["insurer"])])

    if params["objective"] == "binary:logistic" or params["objective"] == "multi:softmax":
        clf = xgb.XGBClassifier(**params)
        clf.compact = False
        pipeline = PMMLPipeline([("classifier", clf)])
    elif params["objective"] =='reg:linear':
        clf = xgb.XGBRegressor(**params)
        clf.compact = False
        pipeline = PMMLPipeline([("estimate", clf)])
    else:
        raise Exception(" Objective function must be in multi:softmax, binary:logistic reg:linear")
        
    if not args["glm_random_split"]:
        pipeline.fit(data.drop(["sample_number", Y_label], axis=1), data[Y_label])
    else:
        pipeline.fit(data.drop(["sample_number", Y_label, args["glm_random_split"]], axis=1), data[Y_label])
    
    to_drop = list(set(['sample_number', 'state', 'Insurer', 'true', 'pred', 'Fold', 'residuals', 'percentile']).intersection(set(preds.columns)))
    if params["objective"] == "binary:logistic" or params["objective"] == "multi:softmax":
        preds["one_fold_preds"] = pipeline.predict_proba(preds.drop(to_drop, axis=1))[:,1]
    elif params["objective"] =='reg:linear':
        preds["one_fold_preds"] = pipeline.predict(preds.drop(to_drop, axis=1))
    else:
        raise Exception(" Objective function must be in multi:softmax, binary:logistic, reg:linear")
        
    sklearn2pmml(pipeline, path_pmml, with_repr= True)
    
    ### check predictions on the same train test split than for GLM
    if args["glm_random_split"]:
        if np.max(data[args["glm_random_split"]]) == 100:
            cap = 80
        elif np.max(data[args["glm_random_split"]]) == 1:
            cap = 0.8
        elif np.max(data[args["glm_random_split"]]) == 10:    
            cap = 8
        else:
            raise Exception(" Please create the random variable such that values are between 0-100, 0-10 or 0-1")
        
        pipeline.fit(data.drop(["sample_number", Y_label, args["glm_random_split"]], axis=1).loc[data[args["glm_random_split"]] <= cap], data[Y_label].loc[data[args["glm_random_split"]] <= cap])

        if params["objective"] == "binary:logistic" or params["objective"] == "multi:softmax":
            preds["glm_random_split"] = pipeline.predict_proba(preds.drop(to_drop + ['one_fold_preds'], axis=1))[:,1]
        elif params["objective"] =='reg:linear':
            preds["glm_random_split"] = pipeline.predict(preds.drop(to_drop + ['one_fold_preds'], axis=1))
        preds[args["glm_random_split"]] = np.where(data[args["glm_random_split"]] <= cap, "train", "test")

    if params["objective"] == "binary:logistic" or params["objective"] == "multi:softmax":
        return preds, pipeline.named_steps['classifier'] #normalizationMethod="logit" 
    else:
        return preds, pipeline.named_steps['estimate']
    
    
def predict_input(args, data, Y_label):
    
    model_path = "/".join([args["path"], args["date"], "02 results", args["cover"], "python_models"])
    models = glob.glob(model_path)
    
    for model in models:
        clf = pickle.load(model)
        if clf.objective == "binary:logistic" or clf.objective == "multi:softmax":
            predictions += clf.predict_proba(data)
    
    return predictions