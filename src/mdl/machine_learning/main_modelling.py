# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:38:04 2017

@author: alexandre
"""

import numpy as np
import pandas as pd
import sys
import os

"""
Done:
    - RMSE error for xgboost
    - Regression XGBOOST
"""

"""
To Do:
    - Linear regression
    - LASSO / Ridge/ elastic net
    - Logistic regression/ LDA
    - Clustering methods
    - Random Forest
    - Keras methods for image recognition
    - Keras methods for text classification
    - GLM/ GAM
    - GMM
    - Bayesian clustering , DBSCAN, KNN, K mean
    - Bum Hunting with PRIM
    - SVM
    - Time series ARMA p, q
"""

class main_modelling(object):
    
    def __ini__(self, global_parameters, data, k_fold = None):
        
        self.global_parameters   = global_parameters
        self.data                = data
        self.k_fold              = k_fold
        
        
    def train(self):
        return
        
    
    def test(self):
        return
    
    
    def cross_validation(self):
        return
    
    
    def load_model(self):
        return
    
    
    def save_model(self):
        return
    
    
    def save_params(self):
        return
    
    
    def create_reports(self):
        return
    
    
    def error_management(self):
        return
    
    
if __name__ == "__main__":
    
    global_parameters = {}
    data = pd.DataFrame()
    
    mm = main_modelling(global_parameters, data)
    
    