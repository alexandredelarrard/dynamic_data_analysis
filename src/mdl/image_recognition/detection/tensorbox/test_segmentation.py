# -*- coding: utf-8 -*-
"""
Created on Mon May 15 16:22:31 2017

@author: alexandre
"""
import tensorflow as tf
from scipy.misc import imresize
import numpy as np
import logging
import os

from segmentation_utils.im_tool_cnn import add_rectangles_test
from segmentation_utils       import googlenet_load
from segmentation_utils.train import build_forward


class Test_tensorbox(object):
    
    def __init__(self, H, pictures_dict, global_parameters, results):
        """
        - Input : H                 : hypes file defined in Main_segmentation function ---> it gives parameters for train and test
                  pictures_dict     : dictionnary of all pictures needed in the back. Dictionnary of PIL pictures for test, list of paths for train
                  results           : dictionnary of results that is filled through the program. It will be split in shape_results in order to create one json per image
                  global_parameters : dictionnary of all parameters needed to make the programe work (family_id, mode, bucket, company, ibm/s3 paths, etc.)
        """
        
        self.H                 = H
        self.pictures_dict     = pictures_dict
        self.results           = results
        self.global_parameters = global_parameters
        self.Test()
        
        
    def Test(self):   
        
        #### Load graph of tensorbox
        type_model = self.global_parameters["required"]["labelID_0"] ### a changer si on fait des segs != categories
        self.H["batch_size"] = 1
        x_in = tf.placeholder(tf.float32, name='x_in', shape=[self.H['image_height'], self.H['image_width'], 3])
            
        googlenet = googlenet_load.init(self.H)
        pred_boxes, pred_logits, pred_confidences = build_forward(self.H, tf.expand_dims(x_in, 0), googlenet, 'test', reuse=None)
        path_to_model = self.global_parameters['model_path'] + "/segmentation_model.ckpt"
    
        ### configure tensorflow session
        config = tf.ConfigProto()    
        config.gpu_options.allocator_type = 'BFC'
        config.log_device_placement=True    
    
        ### launch session and initialize values in graph
        with tf.Session(config = config) as sess:
    
            sess.run(tf.initialize_all_variables())
            tf.train.Saver().restore(sess, path_to_model)
    
            #### type model = "product" or "price"
            #### test_image_name is the name of the picture to segment
            #### idbox is the identity of the segmented box
            for test_image_name in self.pictures_dict.keys():
                self.check_results(test_image_name)
                
                if type_model not in self.results[test_image_name].keys():
                    self.results[test_image_name][type_model] = {}
                    
                old_img = np.array(self.pictures_dict[test_image_name])                
                new_img = imresize(old_img, (self.H["image_height"], self.H["image_width"]), interp='cubic')
    
                # For each croped images
                (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict={x_in: new_img})    
                img_with_rect, rects = add_rectangles_test(self.H, old_img, new_img, np_pred_confidences, np_pred_boxes, use_stitching=True, rnn_len=self.H['rnn_len'], min_conf=self.H["min_conf"], tau=0.2)

                # create final output
                for idbox, r in enumerate(rects):
                    self.results[test_image_name][type_model][idbox] = {"y1" : int(r.y1), 
                                                                        "y2" : int(r.y2), 
                                                                        "x1" : int(r.x1), 
                                                                        "x2" : int(r.x2)}      
                                                                        
                self.check_zero_boxes(type_model, test_image_name)
        
        
    def check_zero_boxes(self, type_model, test_image_name):
        ##### check if any test_image has zero boxe ---> if so, the big picture is given as a box
        print("[SEGMENTATION] [%s] -- origin image name: %20s | Number of boxes: %3d" % (type_model, test_image_name, len(self.results[test_image_name][type_model])))
        if len(self.results[test_image_name][type_model]) == 0:
            logging.error("[SEGMENTATION] [%s] --- 0 box detected " %type_model)
            self.results[test_image_name][type_model] = {}
    
    
    def check_results(self, new_key):
        if new_key not in self.results.keys():
            self.results[new_key] = {}

            
            
            