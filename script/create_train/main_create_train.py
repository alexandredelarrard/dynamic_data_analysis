# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:16:28 2018

@author: JARD
"""
import os
os.environ["DATA_PATH"] = r"C:\Users\User\Documents\tennis\data"

import warnings
warnings.filterwarnings("ignore")

from create_train.data_creation.main_create_history import create_history
from create_train.data_update.main_create_update import create_update


def main_create_train(params):

    create_history(params["rebuild"])
    create_update(params["update"])

    
if __name__ == "__main__":
    params = {"rebuild" : False,
              "update"  : True}
    
    full_data = main_create_train(rebuild=True)
    
    
    