# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:56:26 2017

@author: alexandre
"""
import os
import logging


def check_client_models(global_parameters, table_i):
    """
    Path_name path_[modul]_[model]_[labelId_0]_[name of the resource]
    - Train : Path_name stops up to the labelId_0 (without the name of resource as it will be defined during training)
    - Test  : Path_name
    global_parameters will then have all necessary paths passed for each thread 
    """
    return globals()["check_models_%s"%global_parameters["mode"].lower()](global_parameters, table_i)


def check_models_train(global_parameters, table_i): 
    ### output the path for train up to family
    return "/".join([global_parameters["ibm_client_path"], 'saved_models', table_i["module"], table_i["method"], table_i["model"], table_i["required"]["labelID_0"]])
        

def check_models_prod(global_parameters, table_i): 
    ##### check image recognition models for product
    logger = logging.getLogger("log1")
    
    if "labelID_0" not in table_i["required"]:
        table_i["required"]["labelID_0"] = ""
        
    family_id = table_i["required"]["labelID_0"]
       
    #### all resources needed to run a specific model 
    if table_i["model"] == "facenet" and family_id != "JCseTp3Z9a": 
        liste_models_to_check = ["model-facenet.ckpt" , "index_shelf.pkl", "index_shelf_svc.pkl", "index_shelf_sgd.pkl", "classes.csv", "parameters.json"]
        
    elif table_i["model"] == "facenet" and family_id == "JCseTp3Z9a":
        liste_models_to_check = ["model-facenet.ckpt"]
        
    elif table_i["method"] == "keras":
        liste_models_to_check = ["classes.json" , "weights_kfold.h5", "model.json"]
        
    elif table_i["module"] == "detection":
        liste_models_to_check = ["segmentation_model.ckpt"]
        
    elif table_i["module"] == "association":
        liste_models_to_check = []
        if table_i["method"] == None:
            table_i["method"] = ""
            
    ### if model is in clients storage ---> cool
    ### otherwise check in qopius storage
    for path_ibm_qopius in ["ibm_client_path", "ibm_qopius_path"] :   
        all_present = check_ibm_qopius(global_parameters, path_ibm_qopius, liste_models_to_check, table_i)
        
        if all_present == True:
            return "/".join([global_parameters[path_ibm_qopius], 'saved_models', table_i["module"], table_i["method"], table_i["model"], family_id]).replace("//","/")

    ### model not in qopius storage nor client storage
    if table_i["module"] == "detection":
        if not os.path.isfile("/".join([global_parameters["ibm_qopius_path"], 'saved_models', "detection", "tensorflow", "tensorbox", "MP4AAl4hHR", "segmentation_model.ckpt"])): ### shampoo family
            logger.info("[Detection][%s][segmentation-model] detection model for shampoo NOT found ----> Process KILLED, no detection model found"%table_i["required"]["labelID_0"])
            return ''
            
        else:
            logger.info("[Detection][%s][segmentation-model] detection model used will be Shampoo from qopius storage"%table_i["required"]["labelID_0"])
            return "/".join([global_parameters["ibm_qopius_path"], 'saved_models', "detection", "tensorflow", "tensorbox", "MP4AAl4hHR"])
    else:
        return ''
            
def check_ibm_qopius(global_parameters, path_ibm_qopius, liste_models_to_check, table_i):
    all_present =True         
    for file_model in liste_models_to_check:
        if not os.path.isfile("/".join([global_parameters[path_ibm_qopius], 'saved_models', table_i["module"], table_i["method"], table_i["model"], table_i["required"]["labelID_0"], file_model])):
            print("[%s][%s][%s] Model not found in %s"%(table_i["module"], table_i["required"]["labelID_0"], file_model, path_ibm_qopius))
            all_present = False
        else:
            print("[%s][%s][%s] Model found in %s"%(table_i["module"], table_i["required"]["labelID_0"], file_model, path_ibm_qopius))
            
    return all_present
            