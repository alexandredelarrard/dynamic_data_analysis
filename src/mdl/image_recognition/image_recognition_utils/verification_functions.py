# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:59:14 2016

@author: Antonin Bertin & Kevin Serru
"""

# imports 
import numpy as np
import os
import pandas as pd
import cv2
import sys
import json
# additional imports

# -
def verif_images(folder_path, files_verification):
	"""
	Determine if images can be opened by OpenCV.
	folder_path: STRING path to data
	files_verification: LIST of LIST [[file_0, good_extension], [file_1, good_extension], ...]
	is_corrupted: BOOLEAN 'the file is corrupted'
	"""
	is_corrupted = False
	for file in files_verification:
		is_image = file[1]
		if is_image:
			img = cv2.imread('/'.join([folder_path, file]))
			if img is None:
				is_corrupted = True
	return is_corrupted

def verif_extension(extension, folder_path):
	""" 
	Verify if files in folder_path directory are according to arg_processing
	extension: LIST of accepted files extensions
	folder_path: STRING path to data
	files_verification: LIST of LIST [[file_0, good_extension], [file_1, good_extension], ...]
	all_valid: BOOLEAN 'all the file have a valid extension'
	"""
	files = next(os.walk(folder_path))[2]
	files_verification = []
	n = 0
	all_valid = False
	
	# check if individual files are valid: their extension is in LIST extension
	for file in files:
             valid_extension = False
             for ext in extension:
          
	        if file[-len(ext):] == ext:
                    
	            valid_extension = True
             if valid_extension:
	    	n += 1
             files_verification.append([file, valid_extension])

	# check if all files are valid
 
	if n == len(files):
             
		all_valid = True
       
	return files_verification, all_valid

def verification_functions(application,arg_processing,filename,folder_path):

    statut_verif=False
    
    # check folder is not empty
    is_empty = verif_empty(folder_path)
    if is_empty:
    	statut_verif = False
    	return statut_verif

    else:
        arg_processing=json.loads(arg_processing)
   	# check extension of all files
        
   	extension = arg_processing['extension']
   	[files_verification, all_valid] = verif_extension(extension, folder_path)
   	# if one file has not the desired extension, the function returns False
       
 	if not all_valid:
   		statut_verif = False
   		return statut_verif
 
	# - check for visionwits application
      
	if application == 'visionwits':
                
		# check uploaded images

		if arg_processing['type'] == 'image':
                       
			is_corrupted = verif_images(folder_path, files_verification)
                      
		if is_corrupted:
			statut_verif = False
		else:
			statut_verif = True

	if application == 'retail_logo':
		statut_verif = True   

	if application == 'strat_global_macro':
		statut_verif = True
	
        
    return statut_verif

def logg(warning_message):
	""" This function prints a warning message when prompted """
	print(warning_message)
	return 0

def verif_empty(folder_path):
	"""
	Verify if folder is empty.
	folder_path: STRING path to data
	is_empty: BOOLEAN 'folder is empty'
	"""

	is_empty = False
	n = len(next(os.walk(folder_path))[2])
	if n == 0:
		is_empty = True
	return is_empty





