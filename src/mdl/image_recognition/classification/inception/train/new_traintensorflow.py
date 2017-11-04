#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This modules takes car of the whole training process of image recognition using Tensorflow and model Inception v3
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import os
import random
import re
import struct
import sys
import tarfile
import pandas as pd
from datetime import datetime
from PIL import Image
from random import shuffle
import logging
import numpy as np
from six.moves import urllib
import shutil

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile

d = os.environ["Q_PATH"] + '/q-engine/qopius_visual/utils'
sys.path.append(d)
import COMMON as adminPaths
from ImageAugmenter import ImageAugmenter

d = os.environ["Q_PATH"] + '/q-engine/qopius_visual/utils/utils_siamese'
sys.path.append(d)
from utils_siam_2 import load_data_siamese_from_paths

from new_train_siamese import new_train_siamese

# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long
MINIMUM_IMAGES_PER_CLASS = 40
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 40
MODEL_INPUT_HEIGHT = 40
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
SEED = 448
MULTIPLY_MARKER = "_copy_"
AUGMENTED_MARKER = "_augmented_"
MAX_VALIDATION_PERCENTAGE = 50
MAX_TEST_PERCENTAGE = 25
ADMIN_KEY = "AdminSuperKey"
IMAGE_EXTENSIONS = ('jpg', 'jpeg', 'JPG', 'JPEG')
NO_LABEL = 'no_label'

VERBOSE = True

# - Exception - #


class ClusterisationError(Exception): pass
class InvalidArgumentError(Exception): pass
class InvalidPathError(Exception): pass

# - TrainTensorflow - #


# params = {
# 'userParentLabels': ['highestParent', ..., 'closestParent'],
# 'userLabels': [],  # user labels to include in the training process
# 'api_key':'',  # can be either a admin/user api_key
# 'path_to_data':'',  # where to go find the data
# 'adminParentLabels': ['highestParent', ..., 'closestParent'],
# 'adminLabels': [],  # admin labels to include in the training process
# 'linkAdminToUser': {adminLabel: userLabel}  # all image data from adminLabel is merged with userLabel
# }

# params = {
# 'labels': [],
# 'api_key': '',
# 'path_to_data': ''
# }


class TrainTensorflow(object):
    """Launch the training of image recognition models using Tensorflow

    This is simple transfer learning with an Inception v3 architecture model which
    displays summaries in TensorBoard.
    The top layer receives as input a 2048-dimensional vector for each image. We
    train a softmax layer on top of this representation. Assuming the softmax layer
    contains N labels, this corresponds to learning N + 2048*N model parameters
    corresponding to the learned biases and weights.

    To use with TensorBoard:
    By default, this script will log summaries to /tmp/retrain_logs directory
    Visualize the summaries with this command:
    tensorboard --logdir /tmp/retrain_logs
    """

    def __init__(self, params):

        print("\n")
        print("="*38)
        print("= INITIALIZATION Trainer Object      =")
        print("="*38)
        print("\n")

        self.status = 'user' ### FOR NOW HARD CODED !!!!

        self.params = params

        self.checkArguments()

        # init data and model paths
        self.initDataPaths()
        self.initModelPaths()

        # training parameters
        self.defaultTraining = {
            'how_many_training_steps': 50,  # How many training steps to run before ending.
            'learning_rate': 0.01,  # How large a learning rate to use when training.
            'test_percentage': 0,  # What percentage of images to use as a test set.
            'validation_percentage': 10,  # What percentage of images to use as a validation set.
            'eval_step_interval': 10,  # How often to evaluate the training results.
            'train_batch_size': 10,  # How many images to train on at a time.
            'test_batch_size': 10,  # How many images to test on at a time. This test set is only used infrequently to verify the overall accuracy of the model.
            'validation_batch_size': 10  # How many images to use in an evaluation batch. This validation set is used much more often than the test set, and is an early indicator of how accurate the model is during training.
        }

        self.setTrainingParameters()

        # image augmentation
        self.defaultAugmentation = {
            'augmented_images': 5,  # number of augmented bottlenecks generated per image and stored in bottlenecks_database
            'regenerate': False  # should we erase previous bottlenecks
        }
        self.setImageAugmentation()

        # image clusterisation
        self.defaultClusterisation = {
            'clusterisation': False
        }
        self.setClusterisation()

        # NOT TESTED YET (17 November 2016) - Controls the distortions used during training
        self.flip_left_right = False                                # Whether to randomly flip half of the training images horizontally.
        self.random_crop = 0                                        # A percentage determining how much of a margin to randomly crop off the training images.
        self.random_scale = 0                                       # A percentage determining how much to randomly scale up the size of the training images by.
        self.random_brightness = 0                                  # A percentage determining how much to randomly multiply the training image input pixels up or down by.

        print("\n")
        print("="*38)
        print("======================  LAUNCHING PROCESS  ======================")
        print("="*38)
        print("\n")

        self.main()

    # ------------------------------------------------------------------------------------


    def initDataPaths(self):
        """Sets up data paths"""
        print("Set Data Paths")

        self.image_dir = {}
        self.bottleneck_dir = {}

        # admin data
        if self.status == 'admin' or (self.adminLabels and self.adminParentLabels):
            # images
            self.image_dir['rootAdminCanvas'] = os.path.join(adminPaths.imCanvas, '/'.join(self.adminParentLabels))
            self.image_dir['rootAdminReference'] = os.path.join(adminPaths.imReference, '/'.join(self.adminParentLabels))
            # bottlenecks
            self.bottleneck_dir['rootAdminCanvas'] = os.path.join(adminPaths.botCanvas, '/'.join(self.adminParentLabels))
            self.bottleneck_dir['rootAdminReference'] = os.path.join(adminPaths.botReference, '/'.join(self.adminParentLabels))

        # user data
        if self.status == 'user':

            self.imCanvas = self.path_to_data + '/Picture_database/im_database/'
            self.imReference = self.path_to_data + '/Picture_database/im_database_reference/'
            self.botCanvas = self.path_to_data + '/bottleneck_database/bottleneck_im_database/'
            self.botReference = self.path_to_data + '/bottleneck_database/bottleneck_im_database_reference/'

            # images
            self.image_dir['rootUserCanvas'] = os.path.join(self.imCanvas, '/'.join(self.userParentLabels))
            #self.image_dir['rootUserReference'] = os.path.join(self.imReference, '/'.join(self.userParentLabels))
            # bottlenecks
            self.bottleneck_dir['rootUserCanvas'] = os.path.join(self.botCanvas, '/'.join(self.userParentLabels))
            #self.bottleneck_dir['rootUserReference'] = os.path.join(self.botReference, '/'.join(self.userParentLabels))


        # remove double /
        for k in self.image_dir:
            self.image_dir[k] = self.image_dir[k].replace('//', '/')
            self.bottleneck_dir[k] = self.bottleneck_dir[k].replace('//', '/')

        # check directories exist
        for k in self.image_dir:
            self.ensure_dir_exists(self.image_dir[k])
            self.ensure_dir_exists(self.bottleneck_dir[k])


    def initModelPaths(self):
        """Sets up model paths"""
        print("Set Models Paths")

        now = datetime.now()
        date_today=now.strftime("%Y-%h-%d-%H-%M")

        # inception base model
        inceptionFileName = DATA_URL.split('/')[-1]
        self.model_dir = adminPaths.inceptionV3
        inceptionFilePath = os.path.join(self.model_dir, inceptionFileName)
        if os.path.exists(inceptionFilePath):
            print("Base model found at {}".format(inceptionFilePath))
        else:
            print("Base model not found at {}".format(inceptionFilePath))
            print("-> It will be downloaded online.")

        self.final_tensor_name = 'final_result'  # The name of the output classification layer in the retrained graph.

        # save models
        if self.status == 'admin':
            modelDir = adminPaths.tensorflowModels + '/'.join(self.adminParentLabels)
        else:
            self.tensorflowModels = self.path_to_data + 'saved_models/tensorflow_models/'
            modelDir = self.tensorflowModels + '/'.join(self.userParentLabels)

        self.ensure_dir_exists(modelDir)
        self.output_graph = modelDir + '/output_graph.pb'  # trained graph.
        self.output_labels = modelDir + '/output_labels.txt'  # trained graph's labels.
        print("Models will be saved at: {}".format(modelDir))

        # summary dashboard
        if self.status == 'admin':
            self.summaries_dir = adminPaths.storage + '/tmp/summary_dashboard/' + '_'.join(self.adminParentLabels) + '_' + repr(date_today)
            if not os.path.exists(adminPaths.storage + '/tmp/summary_dashboard/'):
                os.makedirs(adminPaths.storage + '/tmp/summary_dashboard/')
            else:
                shutil.rmtree(adminPaths.storage + '/tmp/summary_dashboard/')
                os.makedirs(adminPaths.storage + '/tmp/summary_dashboard/')

        else:
            self.summaries_dir = self.path_to_data + '/tmp/summary_dashboard/' + '_'.join(self.userParentLabels) + '_' + repr(date_today)
            if not os.path.exists(self.path_to_data + '/tmp/summary_dashboard/'):
                os.makedirs(self.path_to_data + '/tmp/summary_dashboard/')
            else:
                shutil.rmtree(self.path_to_data + '/tmp/summary_dashboard/')
                os.makedirs(self.path_to_data + '/tmp/summary_dashboard/')


    def setTrainingParameters(self):
        """Sets up training parameters:
        - how_many_training_steps
        - learning_rate
        - test_percentage
        - validation_percentage
        - eval_step_interval
        - train_batch_size
        - test_batch_size
        - validation_batch_size
        """
        print("Set Training Parameters")

        self.how_many_training_steps = self.params['how_many_training_steps'] if 'how_many_training_steps' in self.params else self.defaultTraining['how_many_training_steps']
        self.learning_rate = self.params['learning_rate'] if 'learning_rate' in self.params else self.defaultTraining['learning_rate']
        self.test_percentage = self.params['test_percentage'] if 'test_percentage' in self.params else self.defaultTraining['test_percentage']
        self.validation_percentage = self.params['validation_percentage'] if 'validation_percentage' in self.params else self.defaultTraining['validation_percentage']
        self.eval_step_interval = self.params['eval_step_interval'] if 'eval_step_interval' in self.params else self.defaultTraining['eval_step_interval']
        self.train_batch_size = self.params['train_batch_size'] if 'train_batch_size' in self.params else self.defaultTraining['train_batch_size']
        self.test_batch_size = self.params['test_batch_size'] if 'test_batch_size' in self.params else self.defaultTraining['test_batch_size']
        self.validation_batch_size = self.params['validation_batch_size'] if 'validation_batch_size' in self.params else self.defaultTraining['validation_batch_size']
        self.siamese_params = self.params['siamese_config'] if 'siamese_config' in self.params else None


        if ('optional_test' not in self.params) or (not self.params['optional_test']):
            self.test_percentage = 0

        # check percentages
        try:
            if self.validation_percentage > MAX_VALIDATION_PERCENTAGE:
                raise InvalidArgumentError, "validation_percentage: {} is too high. Resetting to: {}".format(self.validation_percentage, MAX_VALIDATION_PERCENTAGE)
            if self.test_percentage > MAX_TEST_PERCENTAGE:
                raise InvalidArgumentError, "test_percentage: {} is too high. Resetting to: {}".format(self.test_percentage, MAX_TEST_PERCENTAGE)

        except InvalidArgumentError, msg:
            print(msg)
            print("Continue...")
            self.validation_percentage = min(MAX_VALIDATION_PERCENTAGE, self.validation_percentage)
            self.test_percentage = min(MAX_TEST_PERCENTAGE, self.test_percentage)


    def setImageAugmentation(self):
        """Sets Image Augmentation Parameters"""
        print("Set Image Augmentation")

        self.augmented_images = self.params['augmented_images'] if 'augmented_images' in self.params else self.defaultAugmentation['augmented_images']
        self.regenerate = self.params['regenerate'] if 'regenerate' in self.params else self.defaultAugmentation['regenerate']

        if self.status == 'admin':
            self.tmp_augmented_images_path = adminPaths.botDatabase + '/tmp_augmentation/'
        else:
            self.tmp_augmented_images_path = self.path_to_data + '/bottleneck_database/tmp_augmentation/'
        self.ensure_dir_exists(self.tmp_augmented_images_path)


    def setClusterisation(self):
        """Sets Clusterisation Parameters"""

        self.clusterisation = self.params['clusterisation'] if 'clusterisation' in self.params else self.defaultClusterisation['clusterisation']

        if self.clusterisation:

            print("Set Clusterisation")

            relativeClusterisationPath = 'Picture_database/clustering/'

            if self.status == 'admin':
                self.pathClusterisationFile = os.path.join(adminPaths.storage, relativeClusterisationPath, 'brand_cluster.csv')
            else:
                self.pathClusterisationFile = os.path.join(self.path_to_data, relativeClusterisationPath, 'brand_cluster.csv')
        try:
            if self.clusterisation and not os.path.exists(self.pathClusterisationFile):
                raise InvalidPathError, "File {} could not be found at path: {}".format(os.path.basename(self.pathClusterisationFile), self.pathClusterisationFile)
        except InvalidPathError, msg:
            print(msg)
            print("- aborting -")
            return 0


    def checkArguments(self):
        """Checks arguments given for training segmentation: mode, params"""

        # params = {
        # 'parentLabels': ['highestParent', ..., 'closestParent'],
        # 'labels': [],  # user labels to include in the training process
        # 'api_key':'',  # can be either a admin/user api_key
        # 'path_to_data':'',  # where to go find the data
        # 'adminParentLabels': ['highestParent', ..., 'closestParent'],
        # 'adminLabels': []  # admin labels to include in the training process
        # }

        print("Check Arguments")
        try:
            # params keys
            for k in ['userParentLabels', 'userLabels', 'api_key', 'path_to_data', 'adminParentLabels', 'adminLabels', 'linkAdminToUser', 'status']:
                if k not in self.params:
                    raise InvalidArgumentError, "Arg: '{}' must contain a key: '{}'".format('params', k)
                else:
                    print("- {}:{}".format(k, repr(self.params[k])))

            # add status param
            self.status = 'admin' if (self.params["api_key"] == ADMIN_KEY or self.params['status'] == 'admin') else 'user'

            for i in self.params:
                exec("self." + i +' = self.params[\''+i+'\']')

        except InvalidArgumentError, msg:
            print(msg)
            print('- aborting -')
            return 0


    # ------------------------------------------------------------------------------------
    def recursive_file_gen(self, d):
        """Recursively walks the arborescence under directory d and returns all the files"""
        for root, dirs, files in os.walk(d):
            for file in files:
                yield os.path.join(root, file)


    def create_augmented_images(self, image, n_images):
        """Gets one image and outputs n_images augmented from image input
        Arguments:
            image {[type]} -- [description]
            n_images {[type]} -- [description]

        Returns:
            [type] -- [description]
        """

        height = image.shape[0]
        width = image.shape[1]
        self.augmenter = ImageAugmenter(
                        width,
                        height, # width and height of the image (must be the same for all images in the batch)
                        hflip=True,    # flip horizontally with 50% probability
                        vflip=False,    # flip vertically with 50% probability
                        scale_to_percent=(0.9,1.1), # scale the image to 70%-130% of its original size
                        scale_axis_equally=True, # allow the axis to be scaled unequally (e.g. x more than y)
                        rotation_deg=(-3,3),    # rotate between -25 and +25 degrees
                        shear_deg=0,       # shear between -10 and +10 degrees
                        translation_x_px=int(width*0.02), # translate between -5 and +5 px on the x-axis
                        translation_y_px=int(height*0.02),  # translate between -5 and +5 px on the y-axis
                        transform_channels_equally=True,
                        jitter = 0  # proportion of pictures having PCA applied
        )
        images, liste_params = self.augmenter.plot_image(image, nb_repeat=n_images)
        return images


    def regenerate_bottlenecks(self, bottleneckLists):
        """Deletes previously stored augmented bottlenecks from the bottlenecks
        database. Note: we only generate augmented bottlenecks for the train section.

        Arguments:
            bottleneckLists = {labelName: {'train': [bottleneckPath],
                                           'validation': [bottleneckPath],
                                           'test': [bottleneckPath]}}
        """
        print("REGENERATION of augmented bottlenecks")
        for labelName, fileLists in bottleneckLists.items():
            for bottleneckPath in fileLists["train"]:
                if (AUGMENTED_MARKER in bottleneckPath) and os.path.exists(bottleneckPath):
                    print("- Removing bottleneck at path: {}".format(bottleneckPath))
                    os.remove(bottleneckPath)


    def PNG_to_JPEG(self, path_to_image):
        """
        This function look for PNG and replace them with JPEGs.
        """
        if path_to_image.endswith(('.png', '.PNG')):
            print(path_to_image)
            f, e = os.path.splitext(os.path.basename(path_to_image))
            safe = True
            try:
                im = Image.open(path_to_image)
            except:
                print("- removing unsafe file {}".format(path_to_image))
                os.remove(path_to_image)
                safe = False

            if safe and (len(np.array(im).shape) == 2 or np.array(im).shape[2] == 4):
                print("- transforming: "+str(path_to_image))
                im.convert('RGB').save('/'.join(path_to_image.split('/')[:-1]+[f+".jpeg"]), 'JPEG')
                os.remove(path_to_image)
                return os.path.join(path_to_image.split('/')[:-1]+[f+".jpeg"])
            else:
                return None
        else:
            return path_to_image


    def links_augmentation(self, imageLists, bottleneckLists, tempImageLists, tempBottleneckLists):
        """
        We want to have at least one element in validation / test sets
        if validation / test have a percentage > 0. (1)
        When the number of available elements in the database is a bit low, the repartition
        between train / validation / test might be a bit tricky.
        E.g: if validation_percentage = 10 / test_percentage = 10 and len(fileList) == 5
        We are going to create as much artificial bottleneck copies as necessary to validate (1)

        Beware, with this function function, some images may be both in train and validation sets. For an alternative that doesn't presents this risk, see: self.dispatchPaths()

        """
        if VERBOSE: print("LINKS AUGMENTATION:")

        for labelName in tempImageLists:

            n0 = int(self.validation_percentage * len(tempImageLists[labelName]) / 100)
            n1 = int(self.test_percentage * len(tempImageLists[labelName]) / 100)
            cond_validation = self.validation_percentage > 0 and len(tempImageLists[labelName][:n0]) == 0
            cond_test = self.test_percentage > 0 and len(tempImageLists[labelName][n0:n0+n1]) == 0
            cond_train = len(tempImageLists[labelName][n0+n1:]) == 0

            # artificially multiply links
            extensionList = [x.split('.', 1)[1] for x in tempImageLists[labelName]]
            extensionBList = [x.split('.', 1)[1] for x in tempBottleneckLists[labelName]]
            tmp = tempImageLists[labelName][:]
            tmpB = tempBottleneckLists[labelName][:]
            i = 0
            while cond_validation or cond_test or cond_train:
                appendList = [x.split('.', 1)[0]+MULTIPLY_MARKER+str(i)+'.'+extensionList[ix] for ix, x in enumerate(tempImageLists[labelName])]
                tmp += appendList
                appendBList = [x.split('.', 1)[0]+MULTIPLY_MARKER+str(i)+'.'+extensionBList[ix] for ix, x in enumerate(tempBottleneckLists[labelName])]
                tmpB += appendBList
                i += 1
                n0 = int(self.validation_percentage * len(tmp) / 100)
                n1 = int(self.test_percentage * len(tmp) / 100)
                cond_validation = self.validation_percentage > 0 and len(tmp[:n0]) == 0
                cond_test = self.test_percentage > 0 and len(tmp[n0:n0+n1]) == 0
                cond_train = len(tmp[n0+n1:]) == 0

            if i > 0:
                # update tempImageLists and tempBottleneckLists
                n = len(tempImageLists[labelName])
                tempImageLists[labelName] = tmp
                tempBottleneckLists[labelName] = tmpB
                if VERBOSE:
                    print("- Augmented Label {:>6} supporting images from {} to {}".format(labelName, n, len(tmp)))


            # shuffle, while keeping the SAME SEED.
                random.seed(SEED)
                shuffle(tempImageLists[labelName])
                random.seed(SEED)
                shuffle(tempBottleneckLists[labelName])


            # get imageLists
            imageLists[labelName] = {
                'train': tempImageLists[labelName][n0+n1:],
                'test': tempImageLists[labelName][n0:n0+n1],
                'validation': tempImageLists[labelName][:n0]
            }

            # get bottleneckLists
            bottleneckLists[labelName] = {
                'train': tempBottleneckLists[labelName][n0+n1:],
                'test': tempBottleneckLists[labelName][n0:n0+n1],
                'validation': tempBottleneckLists[labelName][:n0]
            }

        return imageLists, bottleneckLists


    def dispatchPaths(self, imageLists, bottleneckLists, tempImageLists, tempBottleneckLists):
        """
        This function dispatches data into train / validation / test
        """
        if VERBOSE: print("LINKS REPARTITION:")

        for labelName in tempImageLists:

            # randomize lists
            random.seed(SEED)
            shuffle(tempImageLists[labelName])
            random.seed(SEED)
            shuffle(tempBottleneckLists[labelName])


            n = len(tempImageLists[labelName])
            n0 = int(self.validation_percentage * n / 100)
            n1 = int(self.test_percentage * n / 100)
            cond_validation = self.validation_percentage > 0 and n0 == 0
            cond_test = self.test_percentage > 0 and n1 > 0 and n1 == 0

            if n > 2:
                # - n - 2 images in train
                # - one image in validation
                # - one image in test
                if cond_validation: n0 = 1
                if cond_test: n1 = 1
            elif n == 2:
                # - one image in train
                # - one image in validation
                # - a copy of validation image in test
                if cond_validation: n0 = 1
                if cond_test and not cond_validation: n1 = 1
            else:  # n == 1
                # - one in train
                # - one copy in validation
                # - one copy in test
                n0 = 0
                n1 = 0


            # get imageLists
            imageLists[labelName] = {
                'train': tempImageLists[labelName][n0+n1:],
                'test': tempImageLists[labelName][n0:n0+n1],
                'validation': tempImageLists[labelName][:n0]
            }

            # get bottleneckLists
            bottleneckLists[labelName] = {
                'train': tempBottleneckLists[labelName][n0+n1:],
                'test': tempBottleneckLists[labelName][n0:n0+n1],
                'validation': tempBottleneckLists[labelName][:n0]
            }

            # build copies for imageLists
            if n > 2:
                continue
            elif n == 2:
                if cond_test and cond_validation:
                    imageLists[labelName]['test'] = [x.split('.')[0]+MULTIPLY_MARKER+str(0)+'.'+x.split('.')[1] for x in imageLists[labelName]['validation']]
            else:  # n == 1
                if cond_validation:
                    imageLists[labelName]['validation'] = [x.split('.')[0]+MULTIPLY_MARKER+str(0)+'.'+x.split('.')[1] for x in imageLists[labelName]['train']]
                if cond_test:
                    imageLists[labelName]['test'] = [x.split('.')[0]+MULTIPLY_MARKER+str(1)+'.'+x.split('.')[1] for x in imageLists[labelName]['train']]

            # build copies for bootleneckLists
            if n > 2:
                continue
            elif n == 2:
                if cond_test and cond_validation:
                    bottleneckLists[labelName]['test'] = [x.split('.')[0]+MULTIPLY_MARKER+str(0)+'.'+x.split('.')[1] for x in bottleneckLists[labelName]['validation']]
            else:  # n == 1
                if cond_validation:
                    bottleneckLists[labelName]['validation'] = [x.split('.')[0]+MULTIPLY_MARKER+str(0)+'.'+x.split('.')[1] for x in bottleneckLists[labelName]['train']]
                if cond_test:
                    bottleneckLists[labelName]['test'] = [x.split('.')[0]+MULTIPLY_MARKER+str(1)+'.'+x.split('.')[1] for x in bottleneckLists[labelName]['train']]

        return imageLists, bottleneckLists


    def applyClusterisation(self, tempImageLists, tempBottleneckLists):
        """
        # if clusterisation is activated, each labelName is split into n >= 1 labelNames called
        # labelName_0, labelName_1, ..., labelName_<n-1>.
        # We thus change tempImageLists and tempBottleneckLists to add these new labels.
        """
        df = pd.read_csv(self.pathClusterisationFile, sep=";")

        clusterImageLists = {}
        clusterBottleneckLists = {}

        for labelName, imagePathList in tempImageLists.items():
            for indexPath, imagePath in enumerate(imagePathList):
                clusterID = ""
                imagePath = imagePath.replace('//', '/')

                c = df[df.path == imagePath].cluster.tolist()

                try:
                    if not c:
                        raise ClusterisationError("Warning: image path {} was not found in clusterisation file and thus could not be associated to a cluster.".format(imagePath))
                    elif len(c) > 1:
                        raise ClusterisationError("Warning: image path {} is associated to more than one cluster.".format(imagePath))
                    else:
                        clusterID = labelName + '_' + str(c[0])

                except ClusterisationError, message:
                    print(message)

                except Exception:
                    import traceback
                    traceback.print_ex()

                # complete lists
                if clusterID not in clusterImageLists:
                    clusterImageLists[clusterID] = []
                    clusterBottleneckLists[clusterID] = []

                clusterImageLists[clusterID].append(imagePath)
                bottleneckPath = tempBottleneckLists[labelName][indexPath]
                clusterBottleneckLists[clusterID].append(bottleneckPath)

        for labels in clusterBottleneckLists.keys():
            if len(clusterBottleneckLists[labels])==0:
                clusterBottleneckLists.pop(labels, None)
                clusterImageLists.pop(labels, None)

        print(" clusterImageLists has " + str(len(clusterImageLists)) + " labels")
        print(" clusterBottleneckLists has " + str(len(clusterBottleneckLists)) + " labels")
        return clusterImageLists, clusterBottleneckLists


    def addFiles(self, tempImageLists, tempBottleneckLists, labelsToAdd, rootImDir, rootBotDir, typeLabel):
        """Updates image and bottleneck path lists"""
        for label in labelsToAdd:
            pathImageDir = os.path.join(rootImDir, label)

     

            if os.path.exists(pathImageDir):
                # add all image paths
                tmp = []
                for p in self.recursive_file_gen(pathImageDir):
                    
                    if 'QOPIUS_STORAGE' not in p : 
                        
                       p = self.PNG_to_JPEG(p)
                       if p and p.endswith(IMAGE_EXTENSIONS) :
                          tmp.append(p)
                          
                if tmp:
                    # Once all paths have been generated, if status == user and we go through admin labels, check linkAdminToUser and store all paths under the right user Label.
                    if self.status == 'user' and typeLabel == 'admin':
                        if label in self.linkAdminToUser:
                            label = self.linkAdminToUser[label]

                    # update tempImageLists and tempBottleneckLists
                    if label not in tempImageLists:
                        tempImageLists[label] = []
                        tempBottleneckLists[label] = []
                    tempImageLists[label] += tmp
                    tempBottleneckLists[label] += [x.replace(rootImDir, rootBotDir) +'.txt' for x in tmp]

        return tempImageLists, tempBottleneckLists


    def filterMinimumLinks(self, tempImageLists, tempBottleneckLists, minimumLinks=15):
        tempImageLists = {label: paths for label, paths in tempImageLists.items() if len(paths) >= minimumLinks}
        tempBottleneckLists = {label: paths for label, paths in tempBottleneckLists.items() if len(paths) >= minimumLinks}

        return tempImageLists, tempBottleneckLists


    def generateTempLists(self):
        """Generate temp list to build images and bottlenecks bases"""

        tempImageLists = {}  # {labelName: []}
        tempBottleneckLists = {}  # {labelName: []}

        for pathName, rootImDir in self.image_dir.items():           
         
                     
         
            if pathName in ['rootUserCanvas', 'rootUserReference'] and self.userLabels:
                labelsToAdd = set(self.userLabels)
                
            elif pathName in ['rootAdminCanvas', 'rootAdminReference'] and self.adminLabels:
                labelsToAdd = set(self.adminLabels)
                
            else:
                # take all available labels
                labelsToAdd = set(d for d in next(os.walk(rootImDir))[1])
   
   
            # add 'no_label' label
            labelsToAdd.add(NO_LABEL)

            # update tempImageLists and tempBottleneckLists
            typeLabel = 'admin' if pathName in ['rootAdminCanvas', 'rootAdminReference'] else 'user'
            

            tempImageLists, tempBottleneckLists = self.addFiles(tempImageLists, tempBottleneckLists, labelsToAdd, rootImDir, self.bottleneck_dir[pathName], typeLabel)

        if 'no_label' in tempImageLists.keys():
            # replace key NO_LABEL by a numerical key: -1
            tempImageLists['-1'] = tempImageLists.pop(NO_LABEL)
            tempBottleneckLists['-1'] = tempBottleneckLists.pop(NO_LABEL)

        return tempImageLists, tempBottleneckLists


    def printSizeWarning(self, tempImageLists):
        """Warnings about number of images necessary for training each labels"""
        for ln in tempImageLists:
            nElement = len(tempImageLists[ln])
            if nElement == 0:
                print('- WARNING: Label {:>6} has NO supporting images.'.format(ln))
            elif nElement > 0 and nElement < 20:
                print('- WARNING: Label {:>6} has only {:>3} (< 20) supporting images, which may cause issues.'.format(ln, nElement))
            elif nElement > MAX_NUM_IMAGES_PER_CLASS:
                print('- WARNING: Label {:>6} has more than {} images. Some images will never be selected.'.format(ln, MAX_NUM_IMAGES_PER_CLASS))


    def addAugmentation(self, imageLists, bottleneckLists):
        """Add augmented images and bottlenecks"""
        if VERBOSE:
            print("IMAGE AUGMENTATION: {} new images / image".format(self.augmented_images))

        if self.augmented_images > 0:
            for labelName in imageLists:
                extensionList = [x.split('.', 1)[1] for x in imageLists[labelName]['train']]
                extensionBList = [x.split('.', 1)[1] for x in bottleneckLists[labelName]['train']]
                tmp = imageLists[labelName]['train'][:]
                tmpB = bottleneckLists[labelName]['train'][:]
                i = 0
                while i < self.augmented_images:
                    appendList = [x.split('.', 1)[0]+AUGMENTED_MARKER+str(i)+'.'+extensionList[ix] for ix, x in enumerate(imageLists[labelName]['train'])]
                    tmp += appendList
                    appendBList = [x.split('.', 1)[0]+AUGMENTED_MARKER+str(i)+'.'+extensionBList[ix] for ix, x in enumerate(bottleneckLists[labelName]['train'])]
                    tmpB += appendBList
                    i += 1
                imageLists[labelName]['train'] = tmp
                bottleneckLists[labelName]['train'] = tmpB

        return imageLists, bottleneckLists


    def create_lists(self,option_filter):
        """Create lists of images and bottlenecks"""
        imageLists = {}
        bottleneckLists = {}

        # generate temp lists
        tempImageLists, tempBottleneckLists = self.generateTempLists()

        # filter labels with less than a minimum images (default: 20)
        if option_filter :
           tempImageLists, tempBottleneckLists = self.filterMinimumLinks(tempImageLists, tempBottleneckLists)

        # apply clusterisation on labels
        if self.clusterisation:
            tempImageLists, tempBottleneckLists = self.applyClusterisation(tempImageLists, tempBottleneckLists)

        # print warnings
        if VERBOSE:
            self.printSizeWarning(tempImageLists)

        # artificially multiply paths
        #imageLists, bottleneckLists = self.links_augmentation(imageLists, bottleneckLists, tempImageLists, tempBottleneckLists)

        # dispatch between train / validation / test
        imageLists, bottleneckLists = self.dispatchPaths(imageLists, bottleneckLists, tempImageLists, tempBottleneckLists)

        # add image augmentation on training images
        #imageLists, bottleneckLists = self.addAugmentation(imageLists, bottleneckLists)

        return imageLists, bottleneckLists


    def get_element_path(self, imageLists, labelName, index, category):
        """"
        Returns a path to an image for a label at the given index.
        Args:
            imageLists: Dictionary of training images for each label.
            label_name: Label string we want to get an image for.
            index: Int offset of the image we want. This will be moduloed by the
            available number of images for the label, so it can be arbitrarily large.
            image_dir: Root folder string of the subfolders containing the training
            images.
            category: Name string of set to pull images from - training, testing, or
            validation.
        Returns:
            File system path string to an image that meets the requested parameters.
        """
        if labelName not in imageLists:
            tf.logging.fatal("FATAL: Label '{}' does not exist".format(labelName))

        if category not in imageLists[labelName]:
            tf.logging.fatal("FATAL: Category '{}' does not exist".format(category))

        if category in imageLists[labelName] and len(imageLists[labelName][category]) == 0:
            tf.logging.fatal("FATAL: Label '{}'' has no images in the category '{}'".format(labelName, category))

        elementPath = None
        if category in imageLists[labelName] and len(imageLists[labelName][category]) > 0:
            mod_index = index % len(imageLists[labelName][category])
            elementPath = imageLists[labelName][category][mod_index]

        return elementPath


    def create_inception_graph(self):
        """"Creates a graph from saved GraphDef file and returns a Graph object.
        Returns:
            Graph holding the trained Inception network, and various tensors we'll be
            manipulating.
        """
        with tf.Session() as sess:
            model_filename = os.path.join(
                    self.model_dir, 'classify_image_graph_def.pb')
            with gfile.FastGFile(model_filename, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (tf.import_graph_def(graph_def, name='', return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME, RESIZED_INPUT_TENSOR_NAME]))
        return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


    def run_bottleneck_on_image(self, sess, image_data, image_data_tensor, bottleneck_tensor):
        """Runs inference on an image to extract the 'bottleneck' summary layer.
        Args:
            sess: Current active TensorFlow Session.
            image_data: String of raw JPEG data.
            image_data_tensor: Input data layer in the graph.
            bottleneck_tensor: Layer before the final softmax.
        Returns:
            Numpy array of bottleneck values.
        """
        bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        return bottleneck_values


    def maybe_download_and_extract(self):
        """Download and extract model tar file.
        If the pretrained model we're using doesn't already exist, this function
        downloads it from the TensorFlow.org website and unpacks it into a directory.
        """
        dest_directory = self.model_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):

            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


    def ensure_dir_exists(self, dir_name):
        """Makes sure the folder exists on disk.
        Args:
            dir_name: Path string to the folder we want to create.
        """
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


    def write_list_of_floats_to_file(self, list_of_floats , file_path):
        """Writes a given list of floats to a binary file.
        Args:
            list_of_floats: List of floats we want to write to a file.
            file_path: Path to a file where list of floats will be stored.
        """

        s = struct.pack('d' * BOTTLENECK_TENSOR_SIZE, *list_of_floats)
        with open(file_path, 'wb') as f:
            f.write(s)


    def read_list_of_floats_from_file(self, file_path):
        """Reads list of floats from a given file.
        Args:
            file_path: Path to a file where list of floats was stored.
        Returns:
            Array of bottleneck values (list of floats).
        """

        with open(file_path, 'rb') as f:
            s = struct.unpack('d' * BOTTLENECK_TENSOR_SIZE, f.read())
            return list(s)


    def get_or_create_bottleneck(self, sess, imageLists, bottleneckLists, labelName, index, image_dir, category, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor):
        """Retrieves or calculates bottleneck values for an image.
        If a cached version of the bottleneck data exists on-disk, return that,
        otherwise calculate the data and save it to disk for future use.
        Args:
            sess: The current active TensorFlow Session.
            imageLists: Dictionary of training images for each label.
            label_name: Label string we want to get an image for.
            index: Integer offset of the image we want. This will be modulo-ed by the
            available number of images for the label, so it can be arbitrarily large.
            image_dir: Root folder string  of the subfolders containing the training
            images.
            category: Name string of which  set to pull images from - training, testing,
            or validation.
            bottleneck_dir: Folder string holding cached files of bottleneck values.
            jpeg_data_tensor: The tensor to feed loaded jpeg data into.
            bottleneck_tensor: The output tensor for the bottleneck values.
        Returns:
            Numpy array of values produced by the bottleneck layer for the image.
        """


        bottleneckPath = self.get_element_path(bottleneckLists, labelName, index, category)

        if bottleneckPath is None:
            return None

        # remove multiply marker to get real path
        bottleneckPath = re.sub(r"%s\d+" % MULTIPLY_MARKER, "", bottleneckPath)

        if not os.path.exists(bottleneckPath):

            self.ensure_dir_exists(os.path.split(bottleneckPath)[0])

            imagePath = self.get_element_path(imageLists, labelName, index, category)

            # remove multiply marker to get real path
            imagePath = re.sub(r"%s\d+" % MULTIPLY_MARKER, "", imagePath)

            # check if image must be augmented
            imageToAugment = False
            if AUGMENTED_MARKER in imagePath:
                # launch augmentation once all augmented images are saved only as bootlenecks in the bottlenecks
                # database
                imageToAugment = True
                imagePath = re.sub(r"%s\d+" % AUGMENTED_MARKER, "", imagePath)

            # check if image exists
            if not gfile.Exists(imagePath):
                tf.logging.fatal('FATAL: File does not exist %s', imagePath)

            # If we want to add augmentation to train,
            # - The first time we cannot find one of the augmented bottlenecks associated with an image "I",
            # it means no augmented bottleneck as yet been created from this image, so we create all of them at once
            # - We first generate aug_images from origin "I" through imageAugmenter
            # - We save these aug_images in a temp directory in /bottleneck_database
            # - We generate all augmented bottlenecks from these aug_images and save them in /bottleneck_datanase
            # - After being processed, each aug_image is erased from the temp directory
            # - Next time we look for one of the augmented bottlenecks from image "I", it'll be found next to the others.
            # ---
            # Sometimes when multiple trains are launched on the same data, we actually find the
            # augmented bottleneck from an image without having generated it. It's a reminiscent file
            # from previous trains, and we might want to erase it to generate new augmented bottlenecks
            # from the same origin image "I". So we do the following:
            # If we want to re-generate augmented bottlenecks (through the attribute self.regenerate == True),
            # we call before creating any bottleneck the function self.regenerate_bottlenecks which is
            # going to erase any bottlenecks "named" up to self.augmented_images (e.g: if during the previous train
            # we generated 10 augment bottlenecks per image, and now want to generate 4, only the first 4 former augment
            # bottlenecks will be erased)
            # However, if self.regenerate == False and we have 10 augment bottlenecks already stored, and we want to
            # generate > 10, we will re-build all the bottlenecks base.

            if self.augmented_images > 0 and imageToAugment:
                image = np.array(Image.open(imagePath))
                augmentedImages = self.create_augmented_images(image, self.augmented_images)
                for i, im in enumerate(augmentedImages):
                    try:
                        # save image, load it, bottleneck it, erase it
                        img = Image.fromarray(im, 'RGB')
                        img.save(os.path.join(self.tmp_augmented_images_path, str(i)+'.jpeg'))
                        image = gfile.FastGFile(os.path.join(self.tmp_augmented_images_path, str(i)+'.jpeg'), 'rb').read()
                        bottleneck_values = self.run_bottleneck_on_image(sess, image, jpeg_data_tensor, bottleneck_tensor)
                        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
                        os.remove(os.path.join(self.tmp_augmented_images_path, str(i)+'.jpeg'))
                    except Exception, e:
                        print("WARNING with image at path: {} when converting to bootleneck -> ERROR: {}".format(imagePath, e))
                        return None

                    tmpBottleneckPath = bottleneckPath.split(AUGMENTED_MARKER, 1)[0]+AUGMENTED_MARKER+str(i)+'.'+bottleneckPath.split('.', 1)[1]
                    with open(tmpBottleneckPath, 'w') as bottleneck_file:
                        print('Creating bottleneck at: {}'.format(tmpBottleneckPath))
                        bottleneck_file.write(bottleneck_string)

            else:
                try:
                    image = gfile.FastGFile(imagePath, 'rb').read()
                    bottleneck_values = self.run_bottleneck_on_image(sess, image, jpeg_data_tensor, bottleneck_tensor)
                    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
                except Exception, e:
                    print("WARNING with data at path: {} when converting to bootleneck -> ERROR: {}".format(imagePath, e))
                    return None

                with open(bottleneckPath, 'w') as bottleneck_file:
                    print('Creating bottleneck at: {}'.format(bottleneckPath))
                    bottleneck_file.write(bottleneck_string)

        try:    
            
            with open(bottleneckPath, 'r') as bottleneck_file:
               bottleneck_string = bottleneck_file.read()
               
            bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
            
        except:
            # remove corrupted bottleneck and return None
            print('- WARNING: corrupted bottleneck at path: {}. Removed.'.format(bottleneckPath))
            try : 
               os.remove(bottleneckPath)
            except : 
                print('nothing to remove')
            return None

        return bottleneck_values


    def cache_bottlenecks(self, sess, imageLists, bottleneckLists, image_dir, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor):
        """Ensures all the training, testing, and validation bottlenecks are cached.
        Because we're likely to read the same image multiple times (if there are no
        distortions applied during training) it can speed things up a lot if we
        calculate the bottleneck layer values once for each image during
        preprocessing, and then just read those cached values repeatedly during
        training. Here we go through all the images we've found, calculate those
        values, and save them off.
        Args:
            sess: The current active TensorFlow Session.
            imageLists: Dictionary of training images for each label.
            image_dir: Root folder string of the subfolders containing the training
            images.
            bottleneck_dir: Folder string holding cached files of bottleneck values.
            jpeg_data_tensor: Input tensor for jpeg data from file.
            bottleneck_tensor: The penultimate output layer of the graph.
        Returns:
            Nothing.
        """

        for pathName, pathDataDir in bottleneck_dir.items():
            self.ensure_dir_exists(pathDataDir)

        how_many_bottlenecks = 0

        for labelName, catDict in imageLists.items():
            for category, fileList in catDict.items():
                for index, unused_basename in enumerate(fileList):
                    self.get_or_create_bottleneck(sess, imageLists, bottleneckLists, labelName, index, image_dir, category, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor)
                    how_many_bottlenecks += 1
                    if how_many_bottlenecks % 100 == 0:
                        print(str(how_many_bottlenecks) + ' bottleneck files already created.')


    def get_random_cached_bottlenecks(self, sess, imageLists, bottleneckLists, how_many, category, bottleneck_dir, image_dir, jpeg_data_tensor, bottleneck_tensor):
        """Retrieves bottleneck values for cached images.
        If no distortions are being applied, this function can retrieve the cached
        bottleneck values directly from disk for images. It picks a random set of
        images from the specified category.
        Args:
            sess: Current TensorFlow Session.
            imageLists: Dictionary of training images for each label.
            how_many: The number of bottleneck values to return.
            category: Name string of which set to pull from - training, testing, or
            validation.
            bottleneck_dir: Folder string holding cached files of bottleneck values.
            image_dir: Root folder string of the subfolders containing the training
            images.
            jpeg_data_tensor: The layer to feed jpeg image data into.
            bottleneck_tensor: The bottleneck output layer of the CNN graph.
        Returns:
            List of bottleneck arrays and their corresponding ground truths.
        """
        class_count = len(imageLists.keys())
        bottlenecks = []
        ground_truths = []
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            labelName = list(imageLists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)

            bottleneck = self.get_or_create_bottleneck(sess, imageLists, bottleneckLists, labelName, image_index, image_dir, category, bottleneck_dir, jpeg_data_tensor, bottleneck_tensor)

            if bottleneck is not None:
                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)

        return bottlenecks, ground_truths


    def get_random_distorted_bottlenecks(self, sess, imageLists, how_many, category, image_dir, input_jpeg_tensor, distorted_image, resized_input_tensor, bottleneck_tensor):
        """Retrieves bottleneck values for training images, after distortions.
        If we're training with distortions like crops, scales, or flips, we have to
        recalculate the full model for every image, and so we can't use cached
        bottleneck values. Instead we find random images for the requested category,
        run them through the distortion graph, and then the full graph to get the
        bottleneck results for each.
        Args:
            sess: Current TensorFlow Session.
            imageLists: Dictionary of training images for each label.
            how_many: The integer number of bottleneck values to return.
            category: Name string of which set of images to fetch - training, testing,
            or validation.
            image_dir: Root folder string of the subfolders containing the training
            images.
            input_jpeg_tensor: The input layer we feed the image data to.
            distorted_image: The output node of the distortion graph.
            resized_input_tensor: The input node of the recognition graph.
            bottleneck_tensor: The bottleneck output layer of the CNN graph.
        Returns:
            List of bottleneck arrays and their corresponding ground truths.
        """
        class_count = len(imageLists.keys())
        bottlenecks = []
        ground_truths = []
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(imageLists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_path = self.get_image_path(imageLists, label_name, image_index, image_dir, category)
            if not gfile.Exists(image_path):
                tf.logging.fatal('File does not exist %s', image_path)
            jpeg_data = gfile.FastGFile(image_path, 'rb').read()
            # Note that we materialize the distorted_image_data as a numpy array before
            # sending running inference on the image. This involves 2 memory copies and
            # might be optimized in other implementations.
            distorted_image_data = sess.run(distorted_image, {input_jpeg_tensor: jpeg_data})
            bottleneck = self.run_bottleneck_on_image(sess, distorted_image_data, resized_input_tensor, bottleneck_tensor)
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
        return bottlenecks, ground_truths


    def should_distort_images(self, flip_left_right, random_crop, random_scale, random_brightness):
        """Whether any distortions are enabled, from the input flags.
        Args:
            flip_left_right: Boolean whether to randomly mirror images horizontally.
            random_crop: Integer percentage setting the total margin used around the
            crop box.
            random_scale: Integer percentage of how much to vary the scale by.
            random_brightness: Integer range to randomly multiply the pixel values by.
        Returns:
            Boolean value indicating whether any distortions should be applied.
        """
        return (flip_left_right or (random_crop != 0) or (random_scale != 0) or (random_brightness != 0))


    def add_input_distortions(self, flip_left_right, random_crop, random_scale, random_brightness):
        """Creates the operations to apply the specified distortions.
        During training it can help to improve the results if we run the images
        through simple distortions like crops, scales, and flips. These reflect the
        kind of variations we expect in the real world, and so can help train the
        model to cope with natural data more effectively. Here we take the supplied
        parameters and construct a network of operations to apply them to an image.
        Cropping
        ~~~~~~~~
        Cropping is done by placing a bounding box at a random position in the full
        image. The cropping parameter controls the size of that box relative to the
        input image. If it's zero, then the box is the same size as the input and no
        cropping is performed. If the value is 50%, then the crop box will be half the
        width and height of the input. In a diagram it looks like this:
        <       width         >
        +---------------------+
        |                     |
        |   width - crop%     |
        |    <      >         |
        |    +------+         |
        |    |      |         |
        |    |      |         |
        |    |      |         |
        |    +------+         |
        |                     |
        |                     |
        +---------------------+
        Scaling
        ~~~~~~~
        Scaling is a lot like cropping, except that the bounding box is always
        centered and its size varies randomly within the given range. For example if
        the scale percentage is zero, then the bounding box is the same size as the
        input and no scaling is applied. If it's 50%, then the bounding box will be in
        a random range between half the width and height and full size.
        Args:
            flip_left_right: Boolean whether to randomly mirror images horizontally.
            random_crop: Integer percentage setting the total margin used around the
            crop box.
            random_scale: Integer percentage of how much to vary the scale by.
            random_brightness: Integer range to randomly multiply the pixel values by.
            graph.
        Returns:
            The jpeg input layer and the distorted result tensor.
        """

        jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
        decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_INPUT_DEPTH)
        decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
        decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
        margin_scale = 1.0 + (random_crop / 100.0)
        resize_scale = 1.0 + (random_scale / 100.0)
        margin_scale_value = tf.constant(margin_scale)
        resize_scale_value = tf.random_uniform(tensor_shape.scalar(), minval=1.0, maxval=resize_scale)
        scale_value = tf.mul(margin_scale_value, resize_scale_value)
        precrop_width = tf.mul(scale_value, MODEL_INPUT_WIDTH)
        precrop_height = tf.mul(scale_value, MODEL_INPUT_HEIGHT)
        precrop_shape = tf.pack([precrop_height, precrop_width])
        precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
        precropped_image = tf.image.resize_bilinear(decoded_image_4d, precrop_shape_as_int)
        precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
        cropped_image = tf.random_crop(precropped_image_3d, [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_DEPTH])
        if flip_left_right:
            flipped_image = tf.image.random_flip_left_right(cropped_image)
        else:
            flipped_image = cropped_image
        brightness_min = 1.0 - (random_brightness / 100.0)
        brightness_max = 1.0 + (random_brightness / 100.0)
        brightness_value = tf.random_uniform(tensor_shape.scalar(), minval=brightness_min, maxval=brightness_max)
        brightened_image = tf.mul(flipped_image, brightness_value)
        distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
        return jpeg_data, distort_result


    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.scalar_summary('stddev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)


    def add_final_training_ops(self, class_count, final_tensor_name, bottleneck_tensor):
        """Adds a new softmax and fully-connected layer for training.
        We need to retrain the top layer to identify our new classes, so this function
        adds the right operations to the graph, along with some variables to hold the
        weights, and then sets up all the gradients for the backward pass.
        The set up for the softmax and fully-connected layers is based on:
        https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
        Args:
            class_count: Integer of how many categories of things we're trying to
            recognize.
            final_tensor_name: Name string for the new final node that produces results.
            bottleneck_tensor: The output of the main CNN graph.
        Returns:
            The tensors for the training and cross entropy results, and tensors for the
            bottleneck input and ground truth input.
        """

        with tf.name_scope('input'):
            bottleneck_input = tf.placeholder_with_default(bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
            ground_truth_input = tf.placeholder(tf.float32, [None, class_count], name='GroundTruthInput')

        # Organizing the following ops as `final_training_ops` so they're easier
        # to see in TensorBoard

        '''
        size_layer = 350
        layer_name = 'pre_final_training_ops'
        with tf.name_scope(layer_name):

            with tf.name_scope('weights'):
                layer_weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, size_layer], stddev=0.001), name='pre_final_weights')
                self.variable_summaries(layer_weights, layer_name + '/weights')
            with tf.name_scope('biases'):
                layer_biases = tf.Variable(tf.zeros([size_layer]), name='pre_final_biases')
                self.variable_summaries(layer_biases, layer_name + '/biases')
            with tf.name_scope('Wx_plus_b'):
                #logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
                logits = tf.add(tf.matmul(bottleneck_input, layer_weights), layer_biases)
                tf.histogram_summary(layer_name + '/pre_activations', logits)

        logits = tf.identity(logits, name=layer_name)
        '''
        
        layer_name = 'final_training_ops'
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                layer_weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001), name='final_weights')
                self.variable_summaries(layer_weights, layer_name + '/weights')
            with tf.name_scope('biases'):
                layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
                self.variable_summaries(layer_biases, layer_name + '/biases')
            with tf.name_scope('Wx_plus_b'):
                logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
                tf.histogram_summary(layer_name + '/pre_activations', logits)

        final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
        tf.histogram_summary(final_tensor_name + '/activations', final_tensor)

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, ground_truth_input)
            with tf.name_scope('total'):
                cross_entropy_mean = tf.reduce_mean(cross_entropy)
            tf.scalar_summary('cross entropy', cross_entropy_mean)

        with tf.name_scope('train'):
            train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy_mean)

        return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)


    def add_evaluation_step(self, result_tensor, ground_truth_tensor):
        """Inserts the operations we need to evaluate the accuracy of our results.
        Args:
            result_tensor: The new final node that produces results.
            ground_truth_tensor: The node we feed ground truth data
            into.
        Returns:
            Nothing.
        """
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(result_tensor, 1), tf.argmax(ground_truth_tensor, 1))
            with tf.name_scope('accuracy'):
                evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('accuracy', evaluation_step)
        return evaluation_step


    def main(self):


        if self.siamese_params['siamese_reco'] == True :

           # Look at the folder structure, and create lists of all the images and respective bottlenecks
           imageLists, bottleneckLists = self.create_lists(option_filter=False)

           #generate structure of data for siamese training
           image_paths, image_classes, index_data_no_label = load_data_siamese_from_paths(
                                                                   family_id = int(self.userParentLabels[0]),
                                                                   brand_id = None,
                                                                   type_picture= self.siamese_params['type_picture'],
                                                                   nb_images_per_class_for_train = self.siamese_params['nb_images_per_class_for_train'],
                                                                   nb_images_no_tags=self.siamese_params['nb_images_no_tags'],
                                                                   imageLists = imageLists
                                                                   )


           # launch the train of siamese
           new_train_siamese(
                            use_data = self.path_to_data,
                            family_id= int(self.userParentLabels[0]),
                            brand_id= None,
                            type_picture= self.siamese_params['type_picture'],
                            nb_images_per_class_for_train=self.siamese_params['nb_images_per_class_for_train'],
                            nb_images_per_class_for_index_building=self.siamese_params['nb_images_per_class_for_index_building'],
                            image_paths= image_paths,
                            image_classes= image_classes,
                            index_data_no_label =index_data_no_label,
                            num_epoch = self.siamese_params['nb_iterations']
                            )

        else :

           logging.error("new_traintensorflow.py --> Start training brand/sku reco")
           # Setup the directory we'll write summaries to for TensorBoard
           if tf.gfile.Exists(self.summaries_dir):
               tf.gfile.DeleteRecursively(self.summaries_dir)
           tf.gfile.MakeDirs(self.summaries_dir)

           # Set up the pre-trained graph.
           logging.error("new_traintensorflow.py -->  Set up the pre-trained graph")
           self.maybe_download_and_extract()

           logging.error("new_traintensorflow.py -->  create_inception_graph")
           graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (self.create_inception_graph())

           logging.error("new_traintensorflow.py --> Look at the folder structure, and create lists of all the images and respective bottlenecks")
           # Look at the folder structure, and create lists of all the images and respective bottlenecks

           imageLists, bottleneckLists = self.create_lists(option_filter=False)
           logging.error("new_traintensorflow.py -->  list created ")

           # Possibliy erase previously created bottlenecks
           if self.regenerate:
              self.regenerate_bottlenecks(bottleneckLists)


           # See if the command-line flags mean we're applying any distortions.
           do_distort_images = self.should_distort_images(self.flip_left_right, self.random_crop, self.random_scale, self.random_brightness)

           # launch session
           with tf.Session() as sess:
              if do_distort_images:
                  # We will be applying distortions, so setup the operations we'll need.
                  distorted_jpeg_data_tensor, distorted_image_tensor = self.add_input_distortions(self.flip_left_right, self.random_crop, self.random_scale, self.random_brightness)
              else:
                  # We'll make sure we've calculated the 'bottleneck' image summaries and
                  # cached them on disk.
                  self.cache_bottlenecks(sess, imageLists, bottleneckLists, self.image_dir, self.bottleneck_dir, jpeg_data_tensor, bottleneck_tensor)

              # When a label_name has no images in categories train / validation, suppress label_name.
              catList = ['train']
              if self.validation_percentage > 0: catList.append('validation')
              if self.test_percentage > 0: catList.append('test')

              # if no labelName left, return without training
              if len(imageLists) == 0:
                  return 0

              # Add the new layer that we'll be training.
              #print("num classes : " + str(len(imageLists.keys())))

              (train_step, cross_entropy, bottleneck_input, ground_truth_input,
               final_tensor) = self.add_final_training_ops(len(imageLists.keys()), self.final_tensor_name, bottleneck_tensor)

              # Create the operations we need to evaluate the accuracy of our new layer.
              evaluation_step = self.add_evaluation_step(final_tensor, ground_truth_input)

              # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
              merged = tf.merge_all_summaries()
              train_writer = tf.train.SummaryWriter(self.summaries_dir + '/train', sess.graph)
              validation_writer = tf.train.SummaryWriter(self.summaries_dir + '/validation')

              # Set up all our weights to their initial default values.
              init = tf.initialize_all_variables()
              sess.run(init)

              # Run the training for as many cycles as requested on the command line.
              logging.error('Number of images used %s' %str(imageLists))

              for i in range(self.how_many_training_steps):
                  
                 # Get a batch of input bottleneck values, either calculated fresh every time
                 # with distortions applied, or from the cache stored on disk.
                 if do_distort_images:
                    train_bottlenecks, train_ground_truth = self.get_random_distorted_bottlenecks(sess, imageLists, self.train_batch_size, 'train', self.image_dir, distorted_jpeg_data_tensor, distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
                 else:
                    train_bottlenecks, train_ground_truth = self.get_random_cached_bottlenecks(sess, imageLists, bottleneckLists, self.train_batch_size, 'train', self.bottleneck_dir, self.image_dir, jpeg_data_tensor, bottleneck_tensor)

                 # Feed the bottlenecks and ground truth into the graph, and run a training
                 # step. Capture training summaries for TensorBoard with the `merged` op.
                 train_summary, _ = sess.run([merged, train_step], feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
                 train_writer.add_summary(train_summary, i)

                 # Every so often, print out how well the graph is training.
                 is_last_step = (i + 1 == self.how_many_training_steps)
                 if (i % self.eval_step_interval) == 0 or is_last_step:
                     
                    train_accuracy, cross_entropy_value = sess.run([evaluation_step, cross_entropy], feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
                    print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i, train_accuracy * 100))
                    print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i, cross_entropy_value))
                    validation_bottlenecks, validation_ground_truth = (self.get_random_cached_bottlenecks(sess, imageLists, bottleneckLists, self.validation_batch_size, 'validation', self.bottleneck_dir, self.image_dir, jpeg_data_tensor, bottleneck_tensor))
                    # Run a validation step and capture training summaries for TensorBoard
                    # with the `merged` op.
                    validation_summary, validation_accuracy = sess.run([merged, evaluation_step], feed_dict={bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                    validation_writer.add_summary(validation_summary, i)
                    print('%s: Step %d: Validation accuracy = %.1f%%' % (datetime.now(), i, validation_accuracy * 100))

              if self.test_percentage > 0:
                # We've completed all our training, so run a final test evaluation on
                # some new images we haven't used before.
                test_bottlenecks, test_ground_truth = self.get_random_cached_bottlenecks(sess, imageLists, bottleneckLists, self.test_batch_size, 'test', self.bottleneck_dir, self.image_dir, jpeg_data_tensor, bottleneck_tensor)
                test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
                print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

              # Write out the trained graph and labels with the weights stored as constants.
              output_graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [self.final_tensor_name])
              with gfile.FastGFile(self.output_graph, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
              with gfile.FastGFile(self.output_labels, 'w') as f:
                f.write('\n'.join(imageLists.keys()) + '\n')



def test():
    """Test function"""
    params = {
        'userParentLabels': ['65010000'],  # highest level on the left
        'userLabels': ['brand0'],  # user labels to include in the training process
        'api_key':'api_key0',  # can be either a admin/user api_key
        'status': 'admin',
        'path_to_data':'/home/ubuntu/clients_storage/test_new_train/api_key0/root_train',  # where to go find the data
        'adminParentLabels': ['test_family'],
        'adminLabels': ['brand0'],  # admin labels to include in the training process
        'linkAdminToUser': {}  # link between admin labels and user labels
    }
    tt = TrainTensorflow(params)
    return 0


if __name__ == '__main__':
    test()
