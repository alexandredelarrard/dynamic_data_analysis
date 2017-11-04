#!/usr/local/bin/python2.7

import argparse as ap
import cv2
import imutils 
import numpy as np
import os
import sys
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
import time
from operator import itemgetter
from collections import OrderedDict
from sklearn import ensemble
from sklearn.metrics import roc_curve, auc, log_loss, accuracy_score, mean_absolute_error

import COMMON as shared_pathes
sys.path.append(shared_pathes.caffe_root+'distribute/python/')

## VARIABLES
#path_to_models='/home/ubuntu/bag-of-words/bag-of-words/saved_models/' # path to models
#test_path='/home/ubuntu/bag-of-words/bag-of-words/test2/' # path to data


def upc_reco_sift(object_dict, img_dict, brand_id, fam_id, use_data):

        # object_dict : [[StartY, EndY, StartX, EndX, filename, idbox], ...]

        start_time=time.time()

        if use_data == None:
           path_to_models = shared_pathes.net_root_path + '/' + str(fam_id) + '/UPC_SIFT/' + str(brand_id) + '/'
        else:
           path_to_models = use_data + '/saved_models/' + str(fam_id) + '/UPC_SIFT/' + str(brand_id) + '/'

        objects = []

        if os.path.exists(path_to_models):

          print '---UPC SIFT model found for brand %s ----' %brand_id

          image_paths = []

          ## LOAD MODELS and order them with their accuracy
          list_models={}

          model_list = os.listdir(path_to_models)

          for ml in model_list:

            list_models[ml]=[float(os.path.splitext(str(ml).split('_')[-1])[0])] # get accuracy of all trained models
            list_models[ml].append(os.path.join(path_to_models, ml))

          list_models_active=OrderedDict(sorted(list_models.items(), key=lambda x: x[1][0],reverse=True)) # sort trained models by accuracy

          # Create feature extraction and keypoint detector objects
          fea_det = cv2.FeatureDetector_create("SIFT")
          des_ext = cv2.DescriptorExtractor_create("SIFT")

          # List where all the descriptors are stored
          des_list = []

          print '--- Time until image load : %s' %(time.time()-start_time)

          for product in range(len(object_dict)):  # for each box

            proportion_y = 0.8    
            proportion_x = 0.8
            pinfo = object_dict[product]            
            im = img_dict[pinfo[4]]   
            img = np.array(im)
            img = img[:, :, ::-1].copy()
            image_height, image_width = img.shape[:2]                   
                
            w = pinfo[3] - pinfo[2]
            h = pinfo[1] - pinfo[0]
            dw = w * (1 - proportion_x)/2
            dh = h * (1 - proportion_y)/2
            
            x1 = int(max(0,pinfo[2]-dw))            
            x2 = int(min(image_width-1, pinfo[3]+dw))
            y1 = int(max(0,pinfo[0]-dh))
            y2 = int(min(image_height-1, pinfo[1]+dh))            
                
            im = img[y1:y2,x1:x2]

            kpts = fea_det.detect(im)
            kpts, des = des_ext.compute(im, kpts)
            des_list.append((pinfo[5], des))  
     
            image_paths.append(pinfo[5])

          print '--- Time after feature extraction : %s' %(time.time()-start_time)
            
          # Stack all the descriptors vertically in a numpy array
          descriptors = des_list[0][1]
          for image_path, descriptor in des_list[0:]:
              descriptors = np.vstack((descriptors, descriptor)) 

          predictions=[]

          for model in list_models_active :

            # Load the classifier, class names, scaler, number of clusters and vocabulary

            clf, classes_names, stdSlr, k, voc = joblib.load("%s" %list_models_active[model][1])

            test_features = np.zeros((len(image_paths), k), "float32")
            for i in xrange(len(image_paths)):
                words, distance = vq(des_list[i][1],voc)
                for w in words:
                   test_features[i][w] += 1

            # Perform Tf-Idf vectorization
            nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
            idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

            # Scale the features
            test_features = stdSlr.transform(test_features)

            predictions.append([classes_names[i] for i in clf.predict(test_features)]) # add the predictions of the model for all images tested

            print '------------'
            print '------------'
            print ' --- TEST MODEL %s ----' %model

            print '--- Time after model %s predictions : %s' %(model,time.time()-start_time)

          ## COMPUTE FINAL PREDICTIONS
          global_pred=[]
          number_images=len(predictions[0])
          predictions=np.array(predictions)

          for u in range(0,number_images): # for each image

            global_pred.append(predictions[:,u])

          final_pred={}        
          
          for image_path, pred in zip(image_paths, global_pred):

            f_pred=list(OrderedDict.fromkeys(pred)) # remove duplicates in predictions given by each model for an image and keep order (the order is given by the confidance we have in each model)
            final_pred[image_path]=f_pred[0:5]

          # construct ouptut
          for product in range(len(object_dict)): 
            pinfo = object_dict[product]
            pinfo = pinfo + [np.array(final_pred[pinfo[5]]), np.array([1,1,1,1,1])] # models do not return confidance for the predictions, so they are set to '1'
            objects.append(pinfo)

        else : # model not found

                print '---UPC SIFT model NOT found for brand %s  ;  RETURN  -1 results for sku recognition ----' %str(brand_id)
                print object_dict
                print len(object_dict)
                
                for product in range(len(object_dict)):
                   pinfo = object_dict[product]
                   pinfo = pinfo + [np.array([-1,-1,-1,-1,-1]), np.array([-1,-1,-1,-1,-1])]
                   print pinfo
              
                   objects.append(pinfo)            

        print '--- Total time : %s' %(time.time()-start_time)

        return objects
