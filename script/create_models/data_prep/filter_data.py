# -*- coding: utf-8 -*-
"""
Created on Sun May 13 09:02:29 2018

@author: User
"""

import pandas as pd
import os

def data_prep_for_modelling(data):
    
    full_data = data.copy()
    
    shape0 = full_data.shape[0]
    full_data = full_data.loc[~pd.isnull(full_data["diff_aces"])]
    print("[0] Suppressed missing values : {} suppressed".format(full_data.shape[0] - shape0))
    
    
    
    
    return data_modelling