# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:38:06 2017

@author: alexandre
"""

import os
import logging

from tensorbox.test_segmentation     import Test_tensorbox
from tensorbox.train_segmentation    import Train_tensorbox


class Main_segmentation(object):
    def __init__(self, table_i):
        
        """
        Input : 
            - results               : dictionnary of results send to EC2. The shape is ....
            - pictures_dict         : dictionnary with 3 main keys : "pictures_train_segmentation" (dictionnary), "pictures_train_facenet" (list of path), "pictures_test" (dict of Pil images) 
            - models_dict           : dictionnary with all different paths necessary for any training (facenet of tensorbox)
        Output:
            - No output, fill results if Test. return Json of train KPI's if train.
            - results order of boxes is y1, y2, x1, x2
        """

        self.results                = table_i['results']
        self.pictures_dict          = table_i["pictures_dict"]
        self.global_parameters      = table_i


    def Train(self):
        if self.global_parameters["model"] == "tensorbox":
            self.Hypes()
            self.params["batch_size"] = int(self.global_parameters["parameters"]["batch_size"])
            self.params["focus_size"] = float(self.global_parameters["parameters"]["focus_size"])
            self.params["validation_percentage"] = float(self.global_parameters["parameters"]["split_train_test"])
            self.params["solver"]["max_iter"] = int(self.global_parameters["parameters"]["max_iter"])
            self.params["solver"]["learning_rate"] = float(self.global_parameters["parameters"]["learning_rate"])
            
            Train_tensorbox(self.params, self.pictures_dict, self.global_parameters)
        else:
            print("model other than Tensorbox is not yet defined")
        return {}
        
   
    def Test(self):   
        if self.global_parameters["model"] == "tensorbox":
            self.Hypes()
            try:
                self.params["min_conf"] = float(self.global_parameters["parameters"]["threshold_boxes"])
            except Exception:
                self.params["min_conf"] = 0.2
                pass
            
            test = Test_tensorbox(self.params, self.pictures_dict, self.global_parameters, self.results)
            return test.results
        else:
            print("model other than Tensorbox is not yet defined")
            return {}
            
        
    def Hypes(self):
        
        model_root_path = os.path.dirname(self.global_parameters['model_path']+ "/" + self.global_parameters["required"]["labelID_0"])
        
        self.params = {
            "deconv": False,
            "region_size": 32,
            "num_lstm_layers": 2,
            "rezoom_h_coords": [
                -0.25,
                0.25
            ],
            "grid_width": 40,
            "save_temporary_dir": model_root_path + "/tmp/models",
            "use_lstm": False,
            "lstm_size": 500,
            "focus_size": 1.3,
            "avg_pool_size": 5,
            "early_feat_channels": 256,
            "grid_height": 30,
            "summary_dir": model_root_path + "/tmp/summary_dashboard/%s"%str(self.global_parameters["date"]),
            "use_rezoom": False,
            "rezoom_w_coords": [
                -0.25,
                0.25
            ],
            "rezoom_change_loss": "center",
            "batch_size": 5,
            "reregress": True,
            "data": {
                "test_idl": model_root_path +"/tmp/json/validation_input.json",
                "truncate_data": False,
                "train_idl": model_root_path + "/tmp/json/train_input.json"
            },
            "num_classes": 2,
            "logging": {
                "save_iter": 1000,
                "display_iter": 50
            },
            "rnn_len": 1,
            "solver": {
                "opt": "RMS",
                "rnd_seed": 1,
                "epsilon": 1e-05,
                "learning_rate": 0.001,
                "max_iter": 10000,
                "use_jitter": True,
                "hungarian_iou": 0.25,
                "weights": "",
                "learning_rate_step": 33000,
                "head_weights": [
                    1.0,
                    0.1
                ]
            },
            "model_name": "segmentation_model",
            "path_to_json": model_root_path + "/tmp/json/",
            "path_to_googlenet" : "/".join([os.environ["Q_PATH"], "qopius_storage", "saved_models", "other"]),
            "clip_norm": 1.0,
            "image_height": 960,
            "image_width": 1280,
            "save_dir": model_root_path,
            "biggest_box_px": 100000,
            "min_conf" : 0.3,
            "validation_percentage" : 0.1
        }
        
        