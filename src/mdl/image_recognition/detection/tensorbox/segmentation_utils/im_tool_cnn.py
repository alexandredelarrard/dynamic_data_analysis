import os
import json
from scipy.misc import imread, imresize
from scipy import misc
from itertools import product
import numpy as np
import cv2
import argparse
from math import *
from annolist import AnnotationLib as al
from rect import Rect
#import Image
from PIL import ImageFile

# return coordinates of uper-right corner of all cropes
'''
def crop_CNN(W_im,H_im,w_crop,h_crop) :

    list_crops_W=[u*w_crop for u in range(0,W_im/w_crop) if u*w_crop !=W_im]
    list_crops_H=[u*h_crop for u in range(0,H_im/h_crop) if u*h_crop !=H_im]

    if (W_im-w_crop) not in list_crops_W :
       list_crops_W.append(W_im-w_crop)

    if (H_im-h_crop) not in list_crops_H :
       list_crops_H.append(H_im-h_crop)
    print 'uuut'
    print list_crops_W
    print list_crops_H
    coordinates_crop = list(product(list_crops_W, list_crops_H))

    return coordinates_crop
'''

def crop_CNN(Hw, Hh, I) :

        # zoom I so that its shape is bigger than the standard size
        x_scale = Hw / float(I.shape[1])
        if x_scale > 1:
            I = imresize(I, (int(ceil(x_scale*I.shape[0])), int(ceil(x_scale*I.shape[1]))))
        y_scale = Hh / float(I.shape[0])
        if y_scale > 1:
            I = imresize(I, (int(ceil(y_scale*I.shape[0])), int(ceil(y_scale*I.shape[1]))))

        Iw = I.shape[1]
        Ih = I.shape[0]

        # compute singular coordinates
        coord_x = [0]
        while coord_x[-1] < (Iw-Hw):
            a = coord_x[-1]
            coord_x.append(a+Hw)
        coord_y = [0]
        while coord_y[-1] < (Ih-Hh):
            a = coord_y[-1]
            coord_y.append(a+Hh)

        # correct last coordinate so that no sub image generated is
        # smaller than standard
        coord_x[-1] = Iw-Hw
        coord_y[-1] = Ih-Hh

        # compute full coordinates
        coords = list(product(coord_x, coord_y))  # [(0, 0), (0, Hh), (0, 2*Hh), ...(Hw, 0), (Hw, Hh), ...]

        return coords, I

# return coordinates of uper-right corner of all cropes with overlapping croppes
def crop_CNN_with_overlapping(W_im,H_im,w_crop,h_crop) :

    list_crops_W=[u*w_crop for u in range(0,W_im/(w_crop*2)) if u*w_crop !=W_im]
    list_crops_H=[u*h_crop for u in range(0,H_im/(h_crop*2)) if u*h_crop !=H_im]

    if (W_im-w_crop) not in list_crops_W :
       list_crops_W.append(W_im-w_crop)

    if (H_im-h_crop) not in list_crops_H :
       list_crops_H.append(H_im-h_crop)

    coordinates_crop = list(product(list_crops_W, list_crops_H))

    return coordinates_crop

## function to create json config file for test data
def create_json_config_test_data(list_path_data,path_json_out):

#### list_path_data : list of path to images to test
#### path_json_out : path to json configuration file to test data
####json_file=path_out +'test_boxes_new.json'

    data_out=[]

    for file_path in list_path_data:
        data_out.append({"image_path":str(file_path),"rects":[]})

    jsonFile = open(path_json_out, "w")
    jsonFile.write(json.dumps(data_out))
    jsonFile.close()
    return 0


## function to gather or suppress boxes that are too overlapping based on non maximum supression from  Malisiewicz et al.
# dict_boxes : dict { image_name { id_box [ [ Y1, Y2, X1,X2],[..]] } }

def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    boxes = np.array(boxes)
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    y1 = boxes[:, 0]
    y2 = boxes[:, 1]
    x1 = boxes[:, 2]
    x2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def merge_boxes(boxes, coord_crops):
    """
    If two boxes placed on top of each over overlap along Y-axis,
    and a crop Y-axis instersect overlapping area, merge boxes.

    Inputs:
    - boxes = [[y1, y2, x1, x2]]
    - coord_crops = [(x1, y1)]

    Output:
    -
    """
    id_boxes_merged = []
    new_boxes = []
    for idbox_1, box_coordinates_1 in enumerate(boxes[:-1]):
        for idbox_2, box_coordinates_2 in enumerate(boxes[idbox_1+1:]):

            b1x1 = box_coordinates_1[2]
            b1x2 = box_coordinates_1[3]
            b1y1 = box_coordinates_1[0]
            b1y2 = box_coordinates_1[1]

            b2x1 = box_coordinates_2[2]
            b2x2 = box_coordinates_2[3]
            b2y1 = box_coordinates_2[0]
            b2y2 = box_coordinates_2[1]

            left = max(b1x1, b2x1)
            right = min(b1x2, b2x2)
            top = max(b1y1, b2y1) - 20
            bottom = min(b1y2, b2y2) +20

            # cond_1: boxes overlap
            # cond_2: any crop Y-axis intersects overlapping area
            # cond_3: boxes have not already been merged
            cond_1 = (right - left) * (bottom - top) > 0 and left < right and top < bottom
            cond_2 = any(((y[1] >= top-20) and (y[1] <= bottom+20)) for y in coord_crops)
            cond_3 = idbox_1 not in id_boxes_merged and idbox_2 not in id_boxes_merged

            if cond_1 and cond_2 and cond_3:

                # merge boxes
                nbx1 = min(b1x1, b2x1)
                nbx2 = max(b1x2, b2x2)
                nby1 = min(b1y1, b2y1)
                nby2 = max(b1y2, b2y2)
                new_boxes.append([nby1, nby2, nbx1, nbx2])
                id_boxes_merged += [idbox_1, idbox_2]

    # remove merge boxes and add new boxes
    final_boxes = [boxes[i] for i in range(len(boxes)) if i not in id_boxes_merged] + new_boxes

    return final_boxes


def test_merge_boxes_2():
    """
    Test function for merge_boxes_2()
    """
    boxes = [[0, 10, 0, 10], [5, 15, 5, 15], [0, 10, 20, 25]]
    coord_crops_1 = [(2, 7), (6, 15)]  # overlap a overlapping area
    coord_crops_2 = [(6, 12)]  # overlap a box but not a overlapping area
    coord_crops_3 = [(4, 20)]  # doesn't overlap anything
    print('LAUNCHING TESTS')
    # assert(merge_boxes_2(boxes, coord_crops_1).sort() == [[0, 15, 0, 15], [0, 10, 20, 25]].sort(), "test failed: overlap a overlapping area")
    # assert(merge_boxes_2(boxes, coord_crops_2) == boxes, "test failed: overlap a box but not a overlapping area")
    # assert(merge_boxes_2(boxes, coord_crops_3) == boxes, "test failed: doesn't overlap anything")
    print('ALL TESTS PASSED')


def add_rectangles_test(H, old_img, orig_image, confidences, boxes, use_stitching=False, rnn_len=1, min_conf=0, tau=0.3):
    '''
    add_rectangle_test take results from tensorbox model and output right boxes.
    Boxes diplayed should respect the following needs:
        - minimum confidence in the box > min_conf which is setted by segmentation_th_find function
        - width and height > 15 pixels
        
        Inputs:
        - H : hypes of the model
        - orig_image : picture input and not resized as an array
        - confidences : list of confidences for each box diplayed by tensorbox
        - boxes: well boxes as Annos
        - use_stistching in order to merge boxes overlapping and very close -> useful when there are crops
        - rnn_len: lenght of recurrent neural network
        - min_conf minimum confidence to respect in order to keep the box
        -tau : stitch if more than 30% of overlapping
        
        Outputs:
        - list of boxes
    '''    
    
    image = np.copy(orig_image)
    num_cells = H["grid_height"] * H["grid_width"]
    boxes_r = np.reshape(boxes, (-1,
                                 H["grid_height"],
                                 H["grid_width"],
                                 rnn_len,
                                 4))
    confidences_r = np.reshape(confidences, (-1,
                                             H["grid_height"],
                                             H["grid_width"],
                                             rnn_len,
                                             H['num_classes']))
    cell_pix_size = H['region_size']
    all_rects = [[[] for _ in range(H["grid_width"])] for _ in range(H["grid_height"])]
    for n in range(rnn_len):
        for y in range(H["grid_height"]):
            for x in range(H["grid_width"]):
                bbox = boxes_r[0, y, x, n, :]
                abs_cx = int(bbox[0]) + cell_pix_size/2 + cell_pix_size * x
                abs_cy = int(bbox[1]) + cell_pix_size/2 + cell_pix_size * y
                w = bbox[2]
                h = bbox[3]
                conf = np.max(confidences_r[0, y, x, n, 1:])
                all_rects[y][x].append(Rect(abs_cx,abs_cy,w,h,conf))

    all_rects_r = [r for row in all_rects for cell in row for r in cell]
    if use_stitching:
        from stitch_wrapper import stitch_rects
        acc_rects = stitch_rects(all_rects, tau)
    else:
        acc_rects = all_rects_r

    pairs=[(acc_rects, (0, 255, 0))]

    for rect_set, color in pairs:
        for rect in rect_set:
            if rect.confidence > min_conf:
                cv2.rectangle(image,
                    (rect.cx-int(rect.width/2), rect.cy-int(rect.height/2)),
                    (rect.cx+int(rect.width/2), rect.cy+int(rect.height/2)),
                    color,
                    2)
                    
    flag_resize = True if (old_img.shape[0] != H["image_height"] or old_img.shape[1] != H["image_width"]) else False
    if flag_resize == True:
        x_scale = old_img.shape[1]/ float(orig_image.shape[1])
        y_scale = old_img.shape[0] / float(orig_image.shape[0])
    else:
        x_scale = 1
        y_scale = 1
    
    #### check if width and height are higher than 20 pixels, and check if area is more than 200 pixels
    rects = []
    for rect in acc_rects:
        if rect.confidence > min_conf and rect.width> 15 and rect.height>15:
           r = al.AnnoRect()
           r.x1 = min(old_img.shape[1], max(0,(rect.cx - rect.width/2.)*x_scale))
           r.x2 = min(old_img.shape[1], max(0,(rect.cx + rect.width/2.)*x_scale))
           r.y1 = min(old_img.shape[0], max(0,(rect.cy - rect.height/2.)*y_scale))
           r.y2 = min(old_img.shape[0], max(0,(rect.cy + rect.height/2.)*y_scale))
           r.score = rect.true_confidence
           
           if (r.x2 - r.x1)> 15 and (r.y2 - r.y1)>15:
                rects.append(r)
    
    return image, rects

    
def PIL2array(img):
    print img.size[0]
    print img.size[1]
    print img.getdata()

    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)

def create_croped_images(reference_data, Hw, Hh):
    """
    Generates cropped images used for segmentation

    Inputs:
    - reference_data = {image_origin_name: image}
    - Hw = INT cropped image width
    - Hh = INT cropped image height

    Ouput:
    - reference_data_cropped = {image_origin_name: {'array_image': array_image, 'coordinates': [(x1, y1)], 'cropped_images': {(x1, y1): cropped_image}}}
    """

    # Default:
    #Hw = 1280
    #Hh = 960

    reference_data_cropped = {}

    for image_origin_name, image in reference_data.items():

        # transform base image into array
        array_image = PIL2array(image)

        # find optimal crop coordinates
        # resize without deformation array_image so it fits in a box of size Hw*Hh
        coord_crops, array_image = crop_CNN(Hw, Hh, array_image)
        print(coord_crops)
        
        # update output
        reference_data_cropped[image_origin_name] = {'array_image': array_image, 'coordinates': coord_crops, 'cropped_images': {}}

        # generate cropped images
        for c in coord_crops:

            # get singular coordinates
            x1 = c[0]
            y1 = c[1]
            x2 = min(c[0]+Hw, array_image.shape[1])
            y2 = min(c[1]+Hh, array_image.shape[0])

            # crop base image
            cropped_image = array_image[y1:y2, x1:x2]

            # resize cropped_image if necessary
            if cropped_image.shape != (Hh, Hw):
                cropped_image = imresize(cropped_image, (Hh, Hw), interp='cubic')

            # update output
            reference_data_cropped[image_origin_name]['cropped_images'][str(c)] = cropped_image

    return reference_data_cropped

