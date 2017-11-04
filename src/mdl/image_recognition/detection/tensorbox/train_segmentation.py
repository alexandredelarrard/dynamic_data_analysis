# -*- coding: utf-8 -*-

"""
Created on Mon May 15 16:22:31 2017

@author: alexandre
"""

from random import shuffle
import logging
import json
from segmentation_utils.train import train

class DataVolumeError(Exception): pass

class Train_tensorbox(object):

    def __init__(self, params, pictures_dict, global_parameters):

        self.H                 = params
        self.pictures_dict     = pictures_dict
        self.global_parameters = global_parameters
        self.main()


    def write_input_json(self, data_out, json_name):
        #### Write input JSON from data dictionary """
        with open(self.H["path_to_json"] + json_name, 'w') as jsonFile:
            jsonFile.write(json.dumps(data_out))


    def writeBatchToJson(self, trainData, validationData):
        ####If the volume of data is enought, write batch of data to JSON"""
        if len(trainData) >= 2 and len(validationData) >= 1:
            self.write_input_json(trainData, 'train_input.json')
            self.write_input_json(validationData, 'validation_input.json')
        else:
            raise DataVolumeError, "Not enought data to train a segmentation model Train : %i, Test : %i !"%(len(trainData), len(validationData))


    def buildBatch(self, tempTotal):
        ####Build train/validation batches from tempAdmin and tempUser"""

        print("Total number of pictures for train and validation is: {}".format(len(tempTotal)))
        shuffle(tempTotal)
        
        n = int(len(tempTotal) * self.H["validation_percentage"])  # limit between train and validation
        
        print(tempTotal)
        return tempTotal[n:], tempTotal[:n]


    def main(self):
        
        #### split data into train / test based on percentage
        trainData, validationData = self.buildBatch(self.pictures_dict)
        self.writeBatchToJson(trainData, validationData)

        print("====   enter the train function of tensorbox  ====")
        print("====         Number of iterations is: %i      ====" %self.H["solver"]["max_iter"])
        print("====             Learning rate: %f            ====" %self.H["solver"]["learning_rate"])
        print("====             Batch size: %i               ====" %self.H["batch_size"])

        self.H['th_opt'] = train(self.H)

        with open(self.H['path_to_json']+'/hypes.json', 'w') as f:
            json.dump(self.H, f, indent=4)

        print("H = %f"%self.H['th_opt'])
