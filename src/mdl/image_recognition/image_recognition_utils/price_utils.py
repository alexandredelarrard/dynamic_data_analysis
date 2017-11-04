# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 19:01:21 2016

@author: Antonin Bertin
"""

# imports
import numpy as np
import re
import collections
import os
import cv2
from copy import deepcopy
os.environ['GLOG_minloglevel'] = '2'
import pandas as pd
from PIL import Image
# import Queue
import sys
from threading import Thread
import operator
import pickle
import time
import math
import itertools
# import modules
import scipy
import scipy.cluster.hierarchy as sch
from skimage import exposure
from pylab import cm
from scipy.misc import toimage
import logging

def zoom_square_image(im, to_size):
    return im.resize((to_size, to_size))

def square_price_boxes(ref_segmentation_price, reference_data):
    """
    Returns squared price boxes in preparation for sending to Google-API
    Inputs:
    - ref_segmentation_price = {image_origin_name: {idbox_price: [y1, y2, x1, x2, index_of_shelf]}}
    - reference_data = {image_origin_name: image_origin}

    Output:
    - relative_ref_segmentation = {image_origin_name: {idbox_price: [squared_price_image, ^y1, ^y2, ^x1, ^x2]}}
    (squared_price_image is a square image containing original singular price box and ^y1, ^y2, ^x1, ^x2 are the box coordinates within squared_price_image)
    """

    relative_ref_segmentation = {}

    for image_origin_name, im_dict in ref_segmentation_price.items():

        if len(im_dict)>0 :

            relative_im_dict = {}

            for idbox_price, box_coords in im_dict.items():
                # build sub_image from box
                box_w = box_coords[3] - box_coords[2]
                box_h = box_coords[1] - box_coords[0]
                sub_image_size = max(box_w, box_h)

                ### put in black and white the picture and increase contrast
                background_sub_image = Image.new("RGB", (sub_image_size, sub_image_size), "white")  # generate white background image
                price_sub_image = reference_data[image_origin_name].crop((box_coords[2], box_coords[0], box_coords[3], box_coords[1]))
                price_sub_image = cv2.cvtColor(np.array(price_sub_image), cv2.COLOR_BGR2GRAY)
                price_sub_image = exposure.rescale_intensity(price_sub_image)
                price_sub_image = exposure.adjust_gamma(price_sub_image, 1.2)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                price_sub_image = toimage(clahe.apply(price_sub_image))

                if sub_image_size == box_h:
                    coords = ((sub_image_size - box_w) / 2, 0)
                    ry1 = 0
                    ry2 = sub_image_size
                    rx1 = (sub_image_size - box_w) / 2
                    rx2 = (sub_image_size + box_w) / 2
                elif sub_image_size == box_w:
                    coords = (0, (sub_image_size - box_h) / 2)
                    ry1 = (sub_image_size - box_h) / 2
                    ry2 = (sub_image_size + box_h) / 2
                    rx1 = 0
                    rx2 = sub_image_size

                background_sub_image.paste(price_sub_image, coords)
                relative_im_dict[idbox_price] = [background_sub_image, ry1, ry2, rx1, rx2]

            relative_ref_segmentation[image_origin_name] = relative_im_dict

        else :
            relative_ref_segmentation[image_origin_name] = {}

    return relative_ref_segmentation


def get_price_image(ref_segmentation_price, relative_ref_segmentation, max_size_output_image=1000):
    """
    From square boxes generated in square_price_boxes, build a unique price_image.
    Inputs:
    - ref_segmentation_price = {image_origin_name: {idbox_price: [y1, y2, x1, x2, index_of_shelf]}}
    - relative_ref_segmentation = {image_origin_name: {idbox_price: [squared_price_image, ^y1, ^y2, ^x1, ^x2]}}
    (squared_price_image is a square image containing original singular price box and ^y1, ^y2, ^x1, ^x2 are the box coordinates within squared_price_image)

    Output:
    - image_concatenation = {image_origin_name: [price_image, {idbox_price: [y1, y2, x1, x2]}}
    (price_image is an image made of all squared price boxes)
    ([y1, y2, x1, x2] are absolute coordinates of price boxes in price_image)
    """
    image_concatenation = {}
    for image_origin_name, im_dict in ref_segmentation_price.items():

        if len(im_dict)>0 :
            price_im_dict = []
            n = len(im_dict)  # nb of boxes

            # find integer k so that k^2 - n is positive and minimal
            k = 0
            while k**2 - n < 0:
                k += 1

            # find size of sub_images
            sub_size = int(math.floor(float(max_size_output_image) / k))
            final_size = sub_size*k

            # finally we build a k*k square grid of square sub_images
            coordx = coordy = [i*sub_size for i in list(range(k))]
            coords = list(itertools.product(coordx, coordy))  # [(0, 0), (0, 100)... (0, k*100), (1, 0), ...]

            # generate blank image on which we will apply the sub images. Square it and zoom it.
            #final_image = zoom_square_image(Image.open(os.path.dirname(os.path.realpath(__file__))+'/blank.jpg'), final_size)
            final_image = Image.new("RGB", (final_size, final_size), "white")  # generate white background image

            # get all sub images. Zoom them accordingly. Modify relative coordinates accordingly.
            images = {}
            for idbox, box_relative_info in relative_ref_segmentation[image_origin_name].items():
                im = box_relative_info[0]
                scale = float(sub_size) / im.size[0]
                images[idbox] = [zoom_square_image(im, sub_size)] + [x * scale for x in box_relative_info[1:]]

            # paste to blank image
            use_coord = 0
            final_coordinates = {}
            for idbox, ajusted_box_relative_info in images.items():
                X = coords[use_coord][0]
                Y = coords[use_coord][1]
                ry1 = ajusted_box_relative_info[1]
                ry2 = ajusted_box_relative_info[2]
                rx1 = ajusted_box_relative_info[3]
                rx2 = ajusted_box_relative_info[4]
                final_image.paste(ajusted_box_relative_info[0], coords[use_coord])
                final_coordinates[idbox] = [Y+ry1, Y+ry2, X+rx1, X+rx2]
                use_coord += 1

            # possibly crop final_imagefn
            if use_coord < len(coords) and use_coord % k == 0:
                last_pixel_column = int((use_coord+1)/k)*sub_size
                final_image = final_image.crop((0, 0, last_pixel_column, final_size))

            price_im_dict = [final_image, final_coordinates]
            image_concatenation[image_origin_name] = price_im_dict

        else :
            image_concatenation[image_origin_name] = []

    final_image.save(os.environ["Q_PATH"] + "/image_prix.jpeg")

    return image_concatenation


def extract_price_from_price_image(recognized_price, image_concatenation, ref_segmentation_price):
    """
    Returns price words with absolute coordinates in origin images.
    Inputs:
    - recognized_price = [[image_origin_name, price_word, [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]], ...]  (points are in indirect sense from top left, /!\ in price_image /!\)
    - image_concatenation = {image_origin_name: [price_image, {idbox_price: [y1, y2, x1, x2]}]}  # ([y1, y2, x1, x2] are absolute coordinates of price boxes /!\ in price_image /!\)
    - ref_segmentation_price = {image_origin_name: {idbox_price: [y1, y2, x1, x2]}}

    Output:
    - price_in_file = {image_origin_name: {idbox_price: [[price_word, [x1, y1, x2, y2], [^x1, ^y1, ^x2, ^y2]]]}}
    --> [x1, y1, x2, y2] are coordinates of idbox_price in image_origin
    --> [^x1, ^y1, ^x2, ^y2] are relative coordinates of price word in idbox_price
    """
    price_in_file = {}

    for i in range(len(recognized_price)):  # for each price_word

        image_origin_name = recognized_price[i][0]
        price_word = recognized_price[i][1]
        pw_x1 = recognized_price[i][2][0][0]
        pw_y1 = recognized_price[i][2][0][1]
        pw_x2 = recognized_price[i][2][2][0]
        pw_y2 = recognized_price[i][2][2][1]

        if not image_origin_name in price_in_file:
            price_in_file[image_origin_name] = {}

        idbox_max_area = None  # select good idbox_price on overlapping area with price box
        relative_coordinates = []  # [^x1, ^y1, ^x2, ^y2] relative coordinates of overlapping surface within the idbox_price
        overlapping_max_area = 0

        for idbox_price, box_coords in image_concatenation[image_origin_name][1].items():

            box_x1 = int(box_coords[2])
            box_y1 = int(box_coords[0])
            box_x2 = int(box_coords[3])
            box_y2 = int(box_coords[1])
            left = max(pw_x1, box_x1)
            right = min(pw_x2, box_x2)
            top = max(pw_y1, box_y1)
            bottom = min(pw_y2, box_y2)

            if left < right and top < bottom:
                # box and price_word are overlapping
                overlapping_area = (right - left) * (bottom - top)
                if overlapping_area > overlapping_max_area:
                    idbox_max_area = idbox_price
                    relative_coordinates = [left - box_x1, top - box_y1, right - box_x1, bottom - box_y1]
                    overlapping_max_area = overlapping_area

        if idbox_max_area:
            idbox_price_coordinates_temp = ref_segmentation_price[image_origin_name][idbox_max_area][0:4]  # [y1, y2, x1, x2]
            idbox_price_coordinates = [idbox_price_coordinates_temp[2], idbox_price_coordinates_temp[0], idbox_price_coordinates_temp[3], idbox_price_coordinates_temp[1]]  # [x1, y1, x2, y2]
            price_word_relative_coordinates = relative_coordinates  # [^x1, ^y1, ^x2, ^y2]
            if not idbox_max_area in price_in_file[image_origin_name]:
                price_in_file[image_origin_name][idbox_max_area] = []
            price_in_file[image_origin_name][idbox_max_area].append([price_word]+[idbox_price_coordinates]+[price_word_relative_coordinates])

    logging.error("associated price word to price box")
    logging.error(price_in_file)
    return price_in_file


def find_absolute_position_text(recognized_text, ref_crop):
    """
    Returns absolute position of text in every original image

    Inputs:
    - recognized_text: [[im_crop_name, word, [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]], ...]  (points are in indirect sense from top left)
    - ref_crop : {im_crop_name: [im_origin_name, Y1, X1]}  (X1 and Y1 are top-left coordinates of cropped image)

    Output:
    - text_in_file : {im_origin_name: [[word, X1, Y1, X2, Y2], ...]}
    """

    text_in_file={}  # dictionary containing all texts detected in each images with their absolute positions.
    # recognized_text is a list of list of all words recognized in all images
    # recognized text has the following structure : { {'filename','word_detected','x1','y1','x2','y2','x3','y3','x4','y4'},{'filename','word_detected','x1','y1','x2','y2','x3','y3','x4','y4'},...}
    # [points are in indirect sense from top left] // filename : name of cropped sub image
    for v_f in range(0,len(recognized_text)): # for each word detected


        info_crop=ref_crop[recognized_text[v_f][0]] # get information about the considered crops: from which file, X and Y positions of the crop
        # info_crop[0] is the name of the initial image not croped nor segmented
        # info_crop[1] is the X of the vertical segmentation
        # info_crop[2] is the Y of the horizontal segmentation



        if info_crop[0] not in text_in_file: text_in_file[info_crop[0]]=[]
        listvar=[]

        listvar.append(recognized_text[v_f][1]) # word detected
        listvar.append(recognized_text[v_f][2][0][0]+info_crop[2]) # first X hedge of the text
        listvar.append(recognized_text[v_f][2][0][1]+info_crop[1]) # first Y hedge of the text
        listvar.append(recognized_text[v_f][2][2][0]+info_crop[2]) # second X hedge of the text
        listvar.append(recognized_text[v_f][2][2][1]+info_crop[1]) # second Y hedge of the text

        text_in_file[info_crop[0]].append(listvar)

    # text_in_file is a dictionary containing all words with their corresponding absolute positions for all images; the key of the dictionary are the initial filenames
    # values :[word, X1 hedge, Y1 hedge,X2 hedge, Y2 hedge, ] (X1,Y1) is top left : (X2, Y2) is down right corners of text box
    return  text_in_file


def associate_price_product(price_in_file, ref_segmentation, price_reference, horizontal_bounds_price):
    """
    Associates prices to products
    Inputs:
    - price_in_file = {image_origin_name: [[price_float, x1, y1, x2, y2], ...]}
    - ref_segmentation = {image_origin_name: {idbox: [y1, y2, x1, x2]}}
    - price_reference = {image_origin_name: {bound price Y's: position} (position set to 'up' or 'down' meaning prices are above or bellow products for each shelf)
    - horizontal_bounds_price = {image_origin_name: [bound price Y's]}  (the Y's are price line y-coordinates)

    Outputs:
    - price_list = {image_origin_name: {idbox: [y1, y2, x1, x2, price]}} (where y1, y2, x1, x2 are absolute coordinates of the price box - not the idbox aka product box)
    - ref_segmentation_price_product = {image_origin_name: {idbox: [y1, y2, x1, x2]}}  # coordinates of idboxes take product/price into account
    """

    price_list = {}
    ref_segmentation_price_product = deepcopy(ref_segmentation)

    for origin_image_name in price_in_file:
        price_list[origin_image_name] = {}

        if len(price_in_file[origin_image_name])>0 and len(horizontal_bounds_price[origin_image_name])>0:

            for i, price_float_info in enumerate(price_in_file[origin_image_name]):  # for each word
                price_float = price_float_info[0]
                price_x1 = price_float_info[1]
                price_y1 = price_float_info[2]
                price_x2 = price_float_info[3]
                price_y2 = price_float_info[4]

                # We already identified general "price lines" and identified them by their y-position (-> horizontal_bounds_price).
                # From the y-position of the word, we identify on which horizontal line the word/price is situated.
                Y_price_center = (price_y1 + price_y2) / 2  # y coordinate of word box center
                Y_price_lines = horizontal_bounds_price[origin_image_name]  # price Y lines
                closest_Y_price_line = min(Y_price_lines, key=lambda x: abs(x - Y_price_center))

                # We then want to know if in this "price line" the price etiquettes are generally
                # above (e.g: logitech mice) or under (e.g: shampoo shelves) the product shelf
                price_position = price_reference[origin_image_name][closest_Y_price_line]  # either 'up' or 'down'

                # We then check each product box associated with our "price line" in order to find
                # which one is linked to our price tag.
                for idbox, box_info in ref_segmentation[origin_image_name].items():

                    box_x1 = box_info[2]
                    box_y1 = box_info[0]
                    box_x2 = box_info[3]
                    box_y2 = box_info[1]

                    #price_list[origin_image_name][idbox]=[]

                    # Empirical Threshold Condition:
                    # A product box and a price tag are linked IF the price tag
                    # is closer to the product box than the size of the product box
                    # divided by four along the vertical axis
                    threshold_Y = np.abs(box_y2 - box_y1) / 4

                    # conditions on coordinates for linking a price box to a product box
                    cond_threshold_up = (price_position == 'up' and abs(box_y1 - price_y2) < threshold_Y)
                    cond_threshold_down = (price_position == 'down' and abs(price_y1 - box_y2) < threshold_Y)
                    cond_x = box_x1 < (price_x1 + price_x2) / 2 < box_x2  # middle of the word (price) inside X border of the product

                    # price word must validate one of the two threshold conditions plus the x_condition
                    if (cond_threshold_up or cond_threshold_down) and cond_x:

                        # if a price was already attributed to the product box, take the lowest price (e.g: discounted products)
                        if idbox in price_list[origin_image_name]:
                            if price_list[origin_image_name][idbox] != []:
                                former_price = price_list[origin_image_name][idbox][4]
                                if price_float < former_price:
                                    price_list[origin_image_name][idbox] = [price_y1, price_y2, price_x1, price_x2, price_float]
                        else:
                            price_list[origin_image_name][idbox] = [price_y1, price_y2, price_x1, price_x2, price_float]

                        # update ref_segmentation_price_product
                        y1 = ref_segmentation_price_product[origin_image_name][idbox][0]
                        y2 = ref_segmentation_price_product[origin_image_name][idbox][1]
                        x1 = ref_segmentation_price_product[origin_image_name][idbox][2]
                        x2 = ref_segmentation_price_product[origin_image_name][idbox][3]
                        ref_segmentation_price_product[origin_image_name][idbox] = [
                            min(y1, price_y1),
                            max(y2, price_y2),
                            min(x1, price_x1),
                            max(x2, price_x2)
                        ]

    return price_list, ref_segmentation_price_product


def highlight_boxes(input_image,boxes,output_filename):

    input_image.save('/home/ubuntu/q-engine/qopius_visual/out_images/prov.jpg', format='JPEG')

    with open('/home/ubuntu/q-engine/qopius_visual/out_images/prov.jpg') as image:

        # Reset the file pointer, so we can read the file again
        image.seek(0)

        im = Image.open(image)
        draw = ImageDraw.Draw(im)
        for box_b in boxes:

                box = [(boxes[box_b][2],boxes[box_b][0]),(boxes[box_b][3],boxes[box_b][0]),(boxes[box_b][3],boxes[box_b][1]),(boxes[box_b][2],boxes[box_b][1])]

                draw.line(box + [box[0]], width=5, fill='#00ff00')

        del draw
        im.save(output_filename)


def find_price_reference(shelf_bound_list_product, shelf_bound_list_price):
    """
    Find if price are below or above products for each shelf

    Inputs:
    - shelf_bound_list_product = {image_name: [y0, y1, y2, y3, ...]}  (the y's are middle product line y-coordinates)
    - shelf_bound_list_price = {image_name: [y0, y1, y2, y3, ...]}  (the y's are middle price line y-coordinates)

    Output:
    - price_reference = {image_name: {bound_price_Y: position}}  # position = STRING 'up' or 'down' meaning prices are above or bellow products for each shelf
    """
    price_ref = {}

    for image_name in shelf_bound_list_product:
        price_ref[image_name] = {}
        for shelf_Y_price in shelf_bound_list_price[image_name]: # for each of the Y shelf price bound
            if len(shelf_bound_list_product[image_name]) > 0:
                closest_productY = min(shelf_bound_list_product[image_name], key=lambda x: abs(x-shelf_Y_price)) # find closest price
                price_ref[image_name][shelf_Y_price] = 'down' if closest_productY <= shelf_Y_price else 'up'

            elif len(shelf_bound_list_product[image_name])==0 and len(shelf_bound_list_price[image_name])>0: # if Y price but no Y product then we set the Y_price to none
                shelf_bound_list_price[image_name] = []

    return price_ref


def get_number_of_shelves(ref_segmentation):
    """
    Get the mean number of box at the vertical of each pixel along X-axis in image.
    This gives an estimator of the number of shelves in the image.
    This number is them used in get_horizontal_bound_image to set up the threshold in fcluster.

    Inputs:
    - ref_segmentation = {imageName: {idbox: [y1, y2, x1, x2]}}

    Output:
    - numberOfShelves = {image_name: n}

    Called in:
    price_utils.py/get_horizontal_bound_object
    """
    def _extendList(l, x1, x2):
        if len(l) < x2:
            l += [0 for i in range(x2 - len(l))]
        for x in range(x1, x2):
            l[x] += 1
        return l

    numberOfShelves = {}
    for imageName, boxes in ref_segmentation.items():
        xAxis = []
        for idbox, coordinates in boxes.items():
            _extendList(xAxis, coordinates[2], coordinates[3])

        xAxis = [x for x in xAxis if x > 0]  # suppress all zeros
        xAxis.sort()
        shortAxis = xAxis[len(xAxis)/2:]  # suppress first half of list in case sparce repartition of products

        numberOfShelves[imageName] = sum(shortAxis) / float(len(shortAxis)) if shortAxis else 0

    return numberOfShelves


def parse( lst , avg_height_box):

    ## generate cluster of product for each shelf and determine their position and out of stoc
    ## get cluster :

    cluster = {}
    id_cluster= 0
    cluster[0] = []
    cluster_index = {}
    cluster_index[0] = []

    for idx,i in enumerate(lst):

        if len(cluster[id_cluster]) < 1:    # the first two values are going directly in

            cluster[id_cluster].append(i)
            cluster[id_cluster] = list(set(cluster[id_cluster]))

            cluster_index[id_cluster].append(idx)
            cluster_index[id_cluster] = list(set(cluster_index[id_cluster]))

            continue

        mean = sum(cluster[id_cluster]) / float(len(cluster[id_cluster]))

        if abs(mean - i) > avg_height_box*0.4 :    # check the "distance"

            id_cluster +=1
            cluster[id_cluster] = []
            cluster_index[id_cluster] = []

        cluster[id_cluster].append(i)
        cluster_index[id_cluster].append(idx)

    y_s = []

    for k in cluster.keys():

       if len(cluster[k]) <=2:

          cluster.pop(k)
          cluster_index.pop(k)

       else:
         y_s.append(int(np.mean(cluster[k])))

    return y_s,cluster_index


def get_horizontal_bound_object(ref_segmentation, option_seg):

    """
    Find shelf position using the position of the box or the etiquette of prices
    Inputs:
    - ref_segmentation = {image_name: {idbox: [y1, y2, x1, x2]}}

    Output:
    - horizontal_bound_product = {image_name: [y0, y1, y2, y3, ...]}  (the y's are middle product line y-coordinates)
    """

    horizontal_bound_object = {}

    all_out_of_stock = {}
    all_info_product = {}

    for image_name in ref_segmentation:

        if len(ref_segmentation[image_name])>=2:

            all_out_of_stock[image_name]={}
            all_info_product[image_name] = {}

#           group_price_final = []
            box_middle_list = [(ref_segmentation[image_name][idbox][0] + ref_segmentation[image_name][idbox][1]) / 2 for idbox in ref_segmentation[image_name]] # list of middle of the boxes for each image
            box_middle_X_list = [(ref_segmentation[image_name][idbox][2] + ref_segmentation[image_name][idbox][3]) / 2 for idbox in ref_segmentation[image_name]] # list of middle of the boxes for each image

            avg_height_box = np.median([abs(ref_segmentation[image_name][idbox][1] - ref_segmentation[image_name][idbox][0]) for idbox in ref_segmentation[image_name]])

            from operator import itemgetter
            id_box_list, sorted_list = zip(*sorted(enumerate(box_middle_list), key=itemgetter(1)))

            horizontal_bound_object[image_name], cluster_index = parse(sorted_list, avg_height_box)

            list_box_all_shelfs = [item for sublist in cluster_index.values() for item in sublist]

            if len(list_box_all_shelfs)>0:

               start_X_shelf = min([ref_segmentation[image_name][id_box][2] for id_box in list_box_all_shelfs])
               end_X_shelf = max([ref_segmentation[image_name][id_box][3] for id_box in list_box_all_shelfs])

            else :
               start_X_shelf = min([ref_segmentation[image_name][idbox][2] for idbox in ref_segmentation[image_name]])
               end_X_shelf = max([ref_segmentation[image_name][idbox][3] for idbox in ref_segmentation[image_name]])


            ## get product on shelf position
            if option_seg == 'product' :

                info_product = {}
                out_of_stock = []

                for keyy in cluster_index :

                    ## true index of product on shelf
                    id_box_clust = [id_box_list[box_id] for box_id in cluster_index[keyy]]

                    #sort position of the box in the shelf
                    id_box_list_X, sorted_list_X = zip(*sorted(enumerate([ box_middle_X_list[u] for u in id_box_clust]), key=itemgetter(1)))

                    ## get true sorted position of products on shelf
                    id_box_list_X = [id_box_clust[u] for u in id_box_list_X]

                    ## get mean width value of box for this shelf
                    avg_width_box_shelf = int(np.median([abs(ref_segmentation[image_name][idbox][3] - ref_segmentation[image_name][idbox][2]) for idbox in id_box_clust]))

                    ## take into account larger box than product
                    avg_width_box_shelf =int(avg_width_box_shelf*0.8)

                    for id_box_on_shelf,id_box in enumerate(id_box_list_X):

                        hole = 0

                        info_product[id_box] = {'Shelf Number' : keyy  , 'Product Position' : id_box_on_shelf }

                        if id_box_on_shelf == 0 :    #first product found on shelf

                           if ref_segmentation[image_name][id_box][2] > avg_width_box_shelf :

                               hole = ref_segmentation[image_name][id_box][2]

                               for m in range(0,len(range(0,hole,avg_width_box_shelf)) - 1 ) :

                                   coord_x1 = start_X_shelf + m*avg_width_box_shelf
                                   coord_x2 = start_X_shelf + (m+1)*avg_width_box_shelf
                                   coord_y1 = ref_segmentation[image_name][id_box][0]
                                   coord_y2 = ref_segmentation[image_name][id_box][1]

                                   out_info = {'y1' :coord_y1 ,'y2' :coord_y2 ,'x1': coord_x1, 'x2' :coord_x2 , 'shelf' : keyy }
                                   out_of_stock.append(out_info)

                        if  id_box_on_shelf  < len(id_box_list_X)-1  :

                            next_id_box = id_box_list_X[id_box_on_shelf + 1]


                            hole =  int(abs(ref_segmentation[image_name][next_id_box][2]  - ref_segmentation[image_name][id_box][3]))
                            added_ref_y1 = ref_segmentation[image_name][next_id_box][0]
                            added_ref_y2 = ref_segmentation[image_name][next_id_box][1]
                            coord_start_x = ref_segmentation[image_name][id_box][3]

                        if id_box_on_shelf  == len(id_box_list_X) -1 :  #last product found on shelf


                            hole =  int(abs(end_X_shelf - ref_segmentation[image_name][id_box][3]))
                            added_ref_y1 = ref_segmentation[image_name][id_box][0]
                            added_ref_y2 = ref_segmentation[image_name][id_box][1]
                            coord_start_x = ref_segmentation[image_name][id_box][3]


                        if hole > avg_width_box_shelf :


                           for m in range(0,len(range(0,hole,avg_width_box_shelf)) - 1 ) :

                               coord_x1 = coord_start_x + m*avg_width_box_shelf
                               coord_x2 = coord_start_x + (m+1)*avg_width_box_shelf
                               coord_y1 = int((ref_segmentation[image_name][id_box][0] + added_ref_y1)/float(2))
                               coord_y2 = int((ref_segmentation[image_name][id_box][1] + added_ref_y2)/float(2))

                               out_info = {'y1' :coord_y1 ,'y2' :coord_y2 ,'x1': coord_x1, 'x2' :coord_x2 , 'shelf' : keyy }
                               out_of_stock.append(out_info)

                all_info_product[image_name] = info_product
                all_out_of_stock[image_name] = out_of_stock

        else:

            horizontal_bound_object[image_name] = []

    return horizontal_bound_object,all_info_product,all_out_of_stock


def merge_box_digit(price_in_file):
    """
    Try to merge all price words within a idbox_price (price etiquette) into one single price_word.

    Inputs:
    - price_in_file = {image_origin_name: {idbox_price: [[price_word, [x1, y1, x2, y2], [^x1, ^y1, ^x2, ^y2]]]}}
    --> [x1, y1, x2, y2] are coordinates of idbox_price in image_origin
    --> [^x1, ^y1, ^x2, ^y2] are relative coordinates of price word in idbox_price

    Outputs:
    - price_in_file = {image_origin_name: [[price_word, x1, y1, x2, y2]]}
    """

    price_in_file_out = {}

    for image_origin_name, price_boxes in price_in_file.items():

        price_in_file_out[image_origin_name] = []
        median= []
        for idbox_price, price_words in price_boxes.items():

            # check price_words
            price_words = [x for x in price_words if x[0].isdigit()]

            if len(price_words) == 0:
                continue

            elif len(price_words) == 1:
                # only one price word in price etiquette
                true_price_word = check_price_word(price_words[0][0], median)
                price_in_file_out[image_origin_name].append([true_price_word] + price_words[0][1])
                median.append(true_price_word)

            else:
                price_words = price_words[0]
                true_price_word = check_price_word(price_words[0], median)
                price_in_file_out[image_origin_name].append([true_price_word] + price_words[1])
                median.append(true_price_word)

    return price_in_file_out


def check_price_word(price_word, median_price):

    def _check_numerical(price_word):
        price_word = price_word.replace(',', '.')
        numerical_price_word = filter(lambda x: x.isdigit(), price_word)
        return numerical_price_word

    def check_virgule(numerical_price_word, median):
        minimum = [np.max(median_price) if len(median_price)>0 else 1000]
        real_retour =  float(numerical_price_word)

        for i in range(1, len(numerical_price_word)):
            value = float(numerical_price_word[:-i] +'.'+ numerical_price_word[-i:])
            diff = abs(value - median)
            if diff < minimum:
                real_retour = value
        return real_retour

    if len(median_price)>2:
        median = np.median(median_price)
    else:
        median = 15

    numerical_price_word = _check_numerical(price_word)
    if '.' in numerical_price_word:
        return float(numerical_price_word)

    elif len(numerical_price_word)==2 or len(numerical_price_word)==1:
        return float(numerical_price_word)
    elif len(numerical_price_word)==3:
        numerical_price_word = check_virgule(numerical_price_word, median)
        return numerical_price_word
    elif len(numerical_price_word) >= 4:
        # last two digits are cents digits
        return float(numerical_price_word[:-2]+'.'+numerical_price_word[-2:])
    else:
        return float(price_word)


def add_info_neighboor(image_upc_in):

    ## function to include in the sku short list the products that are detected in the neighboordhood of each box
    # image_upc = {filename: {idbox: [StartY, EndY, StartX, EndX, [brand_id_list], [brand_id_confidence_list], [upc_list], [upc_confidence_list]]}}
    # horizontal_bounds : dict { image:list(set of upper Y boxes bounds) ,... }
    # ref_segmentation : dict() of dict: {filename:{idbox:[StartY, EndY, StartX, EndX, index_of_shelf]}}

    image_upc_out = {}

    for im in image_upc_in: # for each image

        image_upc_out[im] = {}

        for idbox in image_upc_in[im]:  # for each box

            image_upc_out[im][idbox]=image_upc_in[im][idbox]

            try: # try to add right box sku recognition

               print int(image_upc_in[im][idbox+1][6][0])

               if int(image_upc_in[im][idbox+1][6][0])!=-1 and image_upc_out[im][idbox][4][0]==image_upc_out[im][idbox+1][4][0] : # same brand and skus have been recognized in right box

                    image_upc_out[im][idbox][6]=np.append(image_upc_out[im][idbox][6], image_upc_in[im][idbox+1][6][0])
                    image_upc_out[im][idbox][7]=np.append(image_upc_out[im][idbox][7], image_upc_in[im][idbox+1][7][0])

            except:

                pass

            try: # try to add left box sku recognition

                if int(image_upc_in[im][idbox-1][6][0])!=-1 and image_upc_out[im][idbox][4]==image_upc_out[im][idbox-1][4] : # same brand and skus have been recognized in left box

                    image_upc_out[im][idbox][6]=np.append(image_upc_out[im][idbox][6], image_upc_in[im][idbox-1][6][0])
                    image_upc_out[im][idbox][7]=np.append(image_upc_out[im][idbox][7], image_upc_in[im][idbox-1][7][0])
            except:
                pass

    return image_upc_out


def Transform_text_reco_output(recognized_text):

    text_in_file = {}
    for img in recognized_text.keys():
        text_in_file[img] = {}
        for id_box in recognized_text[img].keys():
            liste_per_box = recognized_text[img][id_box]

            words = []
            for i in range(len(liste_per_box)):
                words.append(liste_per_box[i][0])

            text_in_file[img][id_box] = [words, liste_per_box[0][1][0], liste_per_box[0][1][1], liste_per_box[0][1][2] - liste_per_box[0][1][0], liste_per_box[0][1][3]- liste_per_box[0][1][1]]

    return text_in_file

