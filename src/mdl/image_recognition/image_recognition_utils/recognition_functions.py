# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 15:13:53 2016

@author: ben
"""

import sys, cv2
import numpy as np
import string
from images_and_nets import test_fullconv, get_five_best
from images_and_masks import correct_picture_ratio
from ..utils.fct_getters import *
import COMMON as shared_pathes
sys.path.append(shared_pathes.caffe_root+'distribute/python/')

import time
from PIL import Image


def brand_recognition(object_dict,img_dict,fam_id,use_data=None):
    
    proportion_x = 0.8
    proportion_y = 0.8    
    
    if use_data == None:
        net_root = shared_pathes.net_root_path
    else:
        net_root = use_data + '/saved_models/'
        
    recognition_network = net_root + fam_id + '/brand_recognition.prototxt'
    recognition_weights = net_root + fam_id + '/brand_recognition.caffemodel'
    
    net_fc_reco = caffe.Net(recognition_network, recognition_weights, caffe.TEST)
    Net_width = 512
    Net_height = 512    
    desired_ratio = float(Net_width) / float(Net_height)    

    for imgfile in img_dict:
        objects = object_dict[imgfile]
        test_img = img_dict[imgfile]
        img = np.array(test_img)
        img = img[:, :, ::-1].copy()
        image_height, image_width = img.shape[:2]
        
        for i in objects:
            pinfo = objects[i]
          
            w = pinfo[3] - pinfo[2]
            h = pinfo[1] - pinfo[0]
            dw = w * (1 - proportion_x)/2
            dh = h * (1 - proportion_y)/2

            x1 = int(min(image_width-1,max(0,pinfo[2]-dw)))            
            x2 = int(min(image_width-1, pinfo[3]+dw))
            y1 = int(min(image_height-1, max(0,pinfo[0]-dh)))
            y2 = int(min(image_height-1, pinfo[1]+dh))            
                
            product_image = img[y1:y2,x1:x2]
            
            # We have found our products                        
            pimg = correct_picture_ratio(product_image, desired_ratio)
            outimg = cv2.resize(pimg, (Net_width, Net_height))
           # cv2.imshow('img',outimg.astype(np.uint8))        
           # cv2.waitKey(0)
            
            # go through net
            score = test_fullconv(net_fc_reco, outimg.copy())  # ['prob'][0]
            score = score['prob'][0]   
            # get best scores
         
            found, proba = get_five_best(score)

            pinfo[4] = np.array([int(u) for u in found])
            pinfo[5] = np.array(proba)
    
    return object_dict        
            
def UPC_recognition(object_dict, img_dict, brand_id, fam_id, use_data=None):
    
    brand_id = int(brand_id)
    proportion_y = 0.8    
    proportion_x = 0.8
    Net_width = 512
    Net_height = 512    
    desired_ratio = float(Net_width) / float(Net_height)  
    
    if use_data == None:
        net_root = shared_pathes.net_root_path
        path_to_data = shared_pathes.data_root_path
    else:
        net_root = use_data + '/saved_models'
        path_to_data= use_data + '/Picture_database'
    
    fam_df = pd.read_csv('/'.join([path_to_data, "general_database_storage", fam_id, "database_%s.csv" % fam_id]), sep=";", dtype={'upc':str, 'brand_id':int})    
    UPC_model = '/'.join([net_root, fam_id, 'UPCs', str(brand_id)+'_UPC.caffemodel'])
    recognition_network = '/'.join([net_root, fam_id, 'UPCs', str(brand_id)+'_UPC.prototxt'])
    
    objects = []

    if os.path.isfile(UPC_model):
        
        net = caffe.Net(recognition_network, UPC_model, caffe.TEST)                
       
        for product in range(len(object_dict)):  # for each box
            
            pinfo = object_dict[product]
            
            test_img = img_dict[pinfo[4]]
            img = np.array(test_img)
            img = img[:, :, ::-1].copy()
            image_height, image_width = img.shape[:2]     
            #now = time.clock()                
                
            w = pinfo[3] - pinfo[2]
            h = pinfo[1] - pinfo[0]
            dw = w * (1 - proportion_x)/2
            dh = h * (1 - proportion_y)/2
            
            x1 = int(max(0,pinfo[2]-dw))            
            x2 = int(min(image_width-1, pinfo[3]+dw))
            y1 = int(max(0,pinfo[0]-dh))
            y2 = int(min(image_height-1, pinfo[1]+dh))            
                
            product_image = img[y1:y2,x1:x2]
            # We have found our products                        
            pimg = correct_picture_ratio(product_image, desired_ratio)
            outimg = cv2.resize(pimg, (Net_width, Net_height))
             
            #cv2.waitKey(0)
            
            # go through net
            score = test_fullconv(net, outimg.copy())  # ['prob'][0]
            score = score['prob'][0]   
            
            # get best scores
            found, probas = get_five_best(score)

            # probas
            proba=[0.0,0.0,0.0,0.0,0.0]           
            proba[0:len(probas)] = probas                
          
            # upc list
            upc_mock_list = [get_upc_from_index(x, brand_id, fam_id, path_to_data+'/general_database_storage', df=fam_df) for x in found]
            upc_list = [x if x else -1 for x in upc_mock_list]  # replace None elements in mock_list with -1
            

            # take best upc
            UPC_found = upc_list[0]
  
            pinfo = pinfo + [np.array(upc_list), np.array(proba)]            
            objects.append(pinfo)
            #now = time.clock()-now
            #print now
            '''
            if UPC_found != -1:

                new_images = get_images_from_upc(UPC_found, fam_id, path_to_data+'/general_database_storage', path_to_data+'/im_database_reference', path_to_data+'/im_database', df=fam_df)
                        
                image = new_images['true'] + new_images['shelf']
       
                if len(image) > 0:
                  
                    raw_input()
                    true_img = cv2.imread(image[0])
                    true_img = cv2.resize(true_img,(1024,1024))
                    cv2.imwrite(str(UPC_found)+'-'+str(brand_id)+'-inshelf.jpg',outimg.astype(np.uint8))       
                    cv2.imwrite(str(UPC_found)+'-'+str(brand_id)+'-wefound.jpg',true_img.astype('uint8'))
                    
                    #cv2.waitKey(0)
            '''
    else:
        print 'no model found for', str(brand_id)
        for product in range(len(object_dict)):
            pinfo = object_dict[product]
            pinfo = pinfo + [np.array([-1,-1,-1,-1,-1]), np.array([-1,-1,-1,-1,-1])]
            objects.append(pinfo)
    
    return objects

    
    
