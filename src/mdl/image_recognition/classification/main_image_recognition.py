# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 18:28:46 2017

@author: alexandre
"""

import json
from facenet.facenet_test_product     import Test_product_facenet
from facenet.facenet_train_product    import Train_product_facenet
from facenet.facenet_test_price       import Test_price_facenet
from facenet.facenet_train_price      import Train_price_facenet
from keras.keras_test                 import Test_keras
from keras.keras_train                import Train_keras


class Main_image_recognition(object):
    def __init__(self, table_i):
        
        self.results                = table_i['results']
        self.pictures_dict          = table_i["pictures_dict"]
        self.global_parameters      = table_i
        
        
    def Test(self):
        self.parameters_model = self.define_test_parameters(self.global_parameters)
        
        if self.global_parameters["model"] == "facenet" and self.global_parameters["required"]["labelID_0"] != "JCseTp3Z9a":
            test = Test_product_facenet(self.pictures_dict, self.global_parameters, self.results, self.parameters_model)
            return test.results
            
        #### price detection model
        elif self.global_parameters["required"]["labelID_0"] == "JCseTp3Z9a":
            test = Test_price_facenet(self.pictures_dict, self.global_parameters, self.results, self.parameters_model)
            return test.results
        
        elif self.global_parameters["method"] == "keras":
            test = Test_keras(self.pictures_dict, self.global_parameters, self.results, self.parameters_model)
            return test.results
            
        else:
            print("not price nor keras/facenet ---> can't proceeed to test")
            return 0
        
        
    def Train(self):
        self.parameters_model = self.define_train_parameters()
        self.parameters_model["image_size"] =  int(self.global_parameters["parameters"]["image_size"])
        self.parameters_model["embedding_size"] =  int(self.global_parameters["parameters"]["embedding_size"])
        self.parameters_model["learning_rate"] =  float(self.global_parameters["parameters"]["learning_rate"])
        self.parameters_model["num_epoch"] =  int(self.global_parameters["parameters"]["num_epoch"])
        self.parameters_model["epoch_size"] =  int(self.global_parameters["parameters"]["epoch_size"])
        self.parameters_model["split_train_test"] =  float(self.global_parameters["parameters"]["split_train_test"])
        
        if self.global_parameters["model"] == "facenet" and self.global_parameters["required"]["labelID_0"] != "JCseTp3Z9a":
            Train_product_facenet(self.global_parameters, self.pictures_dict, self.parameters_model)
            
        if self.global_parameters["model"] == "facenet" and self.global_parameters["required"]["labelID_0"] == "JCseTp3Z9a":
            Train_price_facenet(self.global_parameters, self.pictures_dict, self.parameters_model)
            
        if self.global_parameters["model"] == "keras":
            Train_keras(self.pictures_dict, self.global_parameters, self.results, self.parameters_model)


    def define_train_parameters(self):

        parameters_model = {
            "batch_size" : 72,
            "image_size" : 160,
            "alpha"      : 0.2,
            "embedding_size"   : 256,
            "keep_probability" : 0.85,
            "weight_decay"     : 0.0001,
            "optimizer"        : "RMSPROP",
            "learning_rate"    : 0.001,
            "learning_rate_decay_epochs"   : 100,
            "learning_rate_decay_factor"   : 0.995,
            "moving_average_decay"         : 0.995,       
            "brand_loss_factor"            : 0.2,
            "center_loss_factor"           : 0.0001,
            "center_loss_alfa"             : 0.85,
            "num_epoch"                    : 20,
            "epoch_size"                   : 1000,
            "seed"                         : 7666,
            "split_train_test"             : 0.15 }
                
        return parameters_model
        

    def define_test_parameters(self, table_i):
        
        if table_i["model"] == "facenet":
            try:
                with open(self.global_parameters['model_path'] + '/parameters.json') as config_file:
                    parameters_model = json.load(config_file)
            except Exception: 
                parameters_model = self.define_train_parameters()
                
        if table_i["method"] == "keras":
             parameters_model = {}
             
        parameters_model["nb_top_upc"]             = 30 
        try:
            parameters_model["threshold_no_label"] = float(table_i["parameters"]["threshold_no_label"])
        except Exception:
            parameters_model["threshold_no_label"] = 0.7
            pass
        
        return parameters_model
        