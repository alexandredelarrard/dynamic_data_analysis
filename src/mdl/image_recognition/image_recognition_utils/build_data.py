# imports
import numpy as np
import collections
import os
import cv2
os.environ['GLOG_minloglevel'] = '2'
import pandas as pd
from PIL import Image
# import Queue
import sys
from threading import Thread
import operator
import pickle
import time



def build_output(im_brand, image_upc, option_reco):
    output_dict = {}

    for filename in im_brand:
        output_dict[filename] = {}

        
        for idbox in im_brand[filename]:

            if option_reco=='text_only':
                output_dict[filename][idbox] = im_brand[filename][idbox]

            if option_reco=='image_only' :
                if image_upc == 0:
                    output_dict[filename][idbox] = im_brand[filename][idbox][0:6]
                else:
                    output_dict[filename][idbox] = image_upc[filename][idbox][0:6] + image_upc[filename][idbox][8:10]

            # joint case
            if option_reco=='text/image':
                if image_upc == 0:
                    output_dict[filename][idbox] = im_brand[filename][idbox]
                else:
                    output_dict[filename][idbox] = image_upc[filename][idbox]
            
    return output_dict
