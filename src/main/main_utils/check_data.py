# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 10:05:46 2017

@author: alexandre
"""

import os
import logging
import zipfile
from PIL import Image, ExifTags
import glob
import json
import numpy as np


def check_client_data(global_parameters, sub_task):
    
    logger = logging.getLogger("log1")
    
    if global_parameters["mode"] == "train":
        im_list=[]
        im_list_client = []
        path_client = "/".join([global_parameters["ibm_client_train_data_path_%s"%sub_task["module"]], sub_task["required"]["labelID_0"]])

        if sub_task["module"] == "detection":

            json_files  = glob.glob(path_client + "/*.json")
            
            for j_son in json_files:
                with open(j_son) as data_file:    
                    dict_json = json.load(data_file)
                    
                for k in range(len(dict_json)):
                    dict_json[k].pop("labels")   
                
                image_path = []    
                for extension in ('jpg', 'jpeg', "png"):
                    image_path.extend(glob.glob("/".join([path_client, os.path.splitext(os.path.basename(j_son))[0] + "*.%s"%extension])))
                
                im_list.append({"rects": dict_json, "image_path" : image_path[0]})
            
            return im_list

        if sub_task["module"] == "classification":
            
            for dirpath, dirnames, filenames in os.walk(path_client):
                    for filename in [f for f in filenames if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".JPEG") or f.endswith(".JPG")]:
                            im_list_client.append(os.path.join(dirpath, filename))

            return im_list_client


    if global_parameters["mode"] == "prod":
        
        pictures_dict = {}
        
        #### open zip file and erase it afterward
        zip_ref = zipfile.ZipFile("/".join([global_parameters["ibm_client_test_data_path"], global_parameters["id_task"] + ".zip"]), 'r')
        zip_ref.extractall(global_parameters["ibm_client_test_data_path"])
        zip_ref.close()

        os.remove("/".join([global_parameters["ibm_client_test_data_path"], global_parameters["id_task"] + ".zip"]))

        #### Rename all files with id of task at first
        ### Rotate pictures when it has to be rotated
        for file in glob.glob(global_parameters["ibm_client_test_data_path"] + "/*"):
            if os.path.splitext(os.path.basename(file))[0][:len(str(global_parameters["id_task"]))] != str(global_parameters["id_task"]):
                new_name = global_parameters["ibm_client_test_data_path"]+'/%s_%s.jpg' %(global_parameters["id_task"], os.path.splitext(os.path.basename(file))[0])
                os.rename(file, new_name)
                file = new_name

            if os.path.getsize(file)>0:
                f, e = os.path.splitext(file)
                image = Image.open(file)
                
                if image.format.upper() in ["JPG", 'JPEG', "PNG"]:
                    file, pictures_dict = check_rotation_image(global_parameters, file, image, pictures_dict, logger)
                    
                elif image.format.upper() not in ["JPG", 'JPEG']:
                    os.remove(file)
                    
        if len(pictures_dict)== 0:
            print("No files can be processed")
        else:
            print("--- Number of files to process is %i" %len(pictures_dict))
            
        return pictures_dict


def check_rotation_image(global_parameters, path_image, image, pictures_dict, logger):
    
    try :
        imIsModified = False
        for orientation in ExifTags.TAGS.keys() :
            if ExifTags.TAGS[orientation]=='Orientation' : break
        try:
            exif = dict(image._getexif().items())
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
                imIsModified = True
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
                imIsModified = True
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
                imIsModified = True
            else:
                pass
        except:
            pass
        
        ### if in PNG then we save as JPEG and reopen with PIL to put it in the pictures dictionnary
        if image.format in ["png", "PNG"]:
            new_path = "/".join([global_parameters["ibm_client_test_data_path"], os.path.splitext(os.path.basename(path_image))[0] + ".jpeg"])
            image.convert('RGB').save(new_path, 'JPEG')
            os.remove(path_image)
            imIsModified = False
            path_image = "/".join([global_parameters["ibm_client_test_data_path"], os.path.splitext(os.path.basename(path_image))[0] + ".jpeg"])
            pictures_dict[os.path.splitext(os.path.basename(path_image))[0]] = np.array(image)[:,:,0:3]
        else:
            pictures_dict[os.path.splitext(os.path.basename(path_image))[0]] = np.array(image)
            
        # save modified images
        if imIsModified:
            image.save(path_image)
            
    except Exception:
        logger.error("FATAL ERROR: image {} could not be loaded.".format(path_image), exc_info=True)
        raise "FATAL ERROR: image {} could not be loaded.".format(path_image)
        
    return path_image, pictures_dict

