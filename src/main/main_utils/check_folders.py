# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 09:37:44 2017

@author: alexandre
"""

import os
import shutil

def check_client_folders(global_parameters, sub_task, i):

    """
    @global_parameters   dictionnary of all parameters needed in the back
    @client              the id of the admin label
    @api_key             the api_key of the model
    @return              { True of False  }
    @base path           path to client folders after api_key level
    """
    
    base_path = global_parameters["ibm_client_path"]

    if i == 0:
        # we check first if api_key is produced
        if not os.path.exists(base_path):
            os.makedirs(base_path)
    
        # then we check if temporary is craeted in order to be able to load json and to get the fam id
        if not os.path.exists(base_path + "/temp"):
            os.makedirs(base_path + "/temp")
            create_temp(base_path+ "/temp", global_parameters)
    
        if not os.path.exists(base_path + "/temp/uploads/"):
            os.makedirs(base_path + "/temp/uploads/")
    
        if not os.path.exists(global_parameters["ibm_client_test_data_path"]):
            os.makedirs(global_parameters["ibm_client_test_data_path"])
    
        if not os.path.exists(global_parameters["ibm_client_temp_output"]):
            os.makedirs(global_parameters["ibm_client_temp_output"])
            
        if not os.path.exists("/".join([base_path, "checked"])):
             os.makedirs("/".join([base_path, "checked"]))
            
        if not os.path.exists("/".join([base_path, "crop"])):
            os.makedirs("/".join([base_path, "crop"]))
        
    #### create arborescence for qopius storage and client storage
    for base_path in [global_parameters["ibm_client_path"], global_parameters["ibm_qopius_path"]]:
        
        if not os.path.exists('/'.join([base_path, "saved_models"])):
            os.makedirs('/'.join([base_path, "saved_models"]))
            
        if "labelID_0" not in sub_task["required"].keys():
            sub_task["required"]["labelID_0"] = ""
    
        client_models_arborescence(base_path, sub_task)
            
        if global_parameters["mode"] == "train":
            check_folder_train(base_path, sub_task)
    
    return global_parameters


def create_temp(base_path, global_parameters):
    os.makedirs(base_path + "/output/")
    os.makedirs(base_path + "/uploads/")
    os.makedirs(global_parameters["ibm_client_test_data_path"])
    os.makedirs(global_parameters["ibm_client_test_config_path"])


def client_models_arborescence(base_path, table_i):
    
    if not os.path.exists('/'.join([base_path,"saved_models", table_i["module"]])):
        os.makedirs('/'.join([base_path, "saved_models", table_i["module"]]))
        
    if not os.path.exists('/'.join([base_path,"saved_models", table_i["module"], table_i["method"]])):
        os.makedirs('/'.join([base_path, "saved_models", table_i["module"], table_i["method"]]))
        
    if not os.path.exists('/'.join([base_path,"saved_models", table_i["module"], table_i["method"], table_i["model"]])):
        os.makedirs('/'.join([base_path, "saved_models", table_i["module"], table_i["method"], table_i["model"]]))
        
    if not os.path.exists('/'.join([base_path,"saved_models", table_i["module"], table_i["method"], table_i["model"], table_i["required"]["labelID_0"]]).replace("//","/")):
        os.makedirs('/'.join([base_path, "saved_models", table_i["module"], table_i["method"], table_i["model"], table_i["required"]["labelID_0"]]).replace("//","/"))

    if not os.path.exists('/'.join([base_path,"saved_models", table_i["module"], table_i["method"], table_i["model"], table_i["required"]["labelID_0"], "tmp"]).replace("//","/")):
        os.makedirs('/'.join([base_path, "saved_models", table_i["module"], table_i["method"], table_i["model"], table_i["required"]["labelID_0"], "tmp"]).replace("//","/"))


def check_folder_train(base_path, table_i):
    
    if not os.path.exists('/'.join([base_path, "checked"]).replace("//","/")):
        os.makedirs('/'.join([base_path, "checked"]).replace("//","/"))
     
    if not os.path.exists('/'.join([base_path, "checked", table_i["required"]["labelID_0"]]).replace("//","/")):
        os.makedirs('/'.join([base_path, "checked", table_i["required"]["labelID_0"]]).replace("//","/"))

    if not os.path.exists('/'.join([base_path, "saved_models", table_i["module"], table_i["method"],  table_i["model"], table_i["required"]["labelID_0"], 'tmp', 'models']).replace("//","/")):
        os.makedirs('/'.join([base_path, "saved_models", table_i["module"], table_i["method"],  table_i["model"], table_i["required"]["labelID_0"], 'tmp', 'models']).replace("//","/"))
        
    if not os.path.exists('/'.join([base_path, "saved_models", table_i["module"], table_i["method"],  table_i["model"], table_i["required"]["labelID_0"], 'tmp', 'json']).replace("//","/")):
        os.makedirs('/'.join([base_path, "saved_models", table_i["module"], table_i["method"],  table_i["model"], table_i["required"]["labelID_0"], 'tmp', 'json']).replace("//","/"))
      
    if not os.path.exists('/'.join([base_path, "saved_models", table_i["module"], table_i["method"],  table_i["model"], table_i["required"]["labelID_0"], 'tmp', 'summary_dashboard']).replace("//","/")):
        os.makedirs('/'.join([base_path, "saved_models", table_i["module"], table_i["method"],  table_i["model"], table_i["required"]["labelID_0"], 'tmp', 'summary_dashboard']).replace("//","/"))
    else:
        shutil.rmtree('/'.join([base_path, "saved_models", table_i["module"], table_i["method"],  table_i["model"], table_i["required"]["labelID_0"], 'tmp', 'summary_dashboard']).replace("//","/"))
        os.makedirs('/'.join([base_path, "saved_models", table_i["module"], table_i["method"],  table_i["model"], table_i["required"]["labelID_0"], 'tmp', 'summary_dashboard']).replace("//","/"))
    
