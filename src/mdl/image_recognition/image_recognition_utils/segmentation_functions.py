# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 15:00:39 2016

@author: ben

Functions developed for image segmentation

TODO: finetune all these annoying parameters

Implements:

 - segmentation(im, mask_loc, mask_seg, save = None)
     Main segmentation function. Uses the Lmask and Smask outputs of a localization and segmentation analysis
     Iteratively serches for shelves, group of products, then each product indepentently
     Each of these functions is parametrized manually
 
 - get_shelf_position(mask)
     sums horizontally the localization mask and looks for start/end of shelves
 - get_group_of_products(mask,start_product_H, end_product_H, H_zones)
     sums vertically in a shelf to find start/end of a group of products
 - get_product_segmentation(mask,start_shelf,end_shelf,start_gop,end_gop)
     sums vertically the segmentation mask to estimate product bounding boxes

 - remove_outlier(borders,nbr_borders)
     eliminates boxes that are too small (less than a percentage of the average product width)

"""
import numpy as np
import cv2
from images_and_masks import show_centers
import sys
from images_and_nets import Im_pyramidal_net_scan, analyze_products, set_pyramid, get_five_best, test_fullconv
from data_manipulation import load_data_xls, save_objects_in_xls, combine_objects, create_imgs_from_objects, compare_boxes_quadratic_distance, compare_boxes
from images_and_masks import show_objects, combine_masks, correct_picture_ratio
import os 
import COMMON as shared_pathes

sys.path.append(shared_pathes.caffe_root+'distribute/python/')

def get_shelf_position(mask):
    h, w = (np.array(mask.shape[:2])).astype(np.int)
    thresold = 0.25
    empiric_min_size = h/15.0
    ## Get horizontal borders
    # compute rawsum of binary mask
    rawsum_bm = 0
    for j in range(w):
        rawsum_bm = rawsum_bm + mask[:,j]

    start_product_H = np.zeros(50)
    end_product_H = np.zeros(50)

    # first line is products?
    if rawsum_bm[0] > thresold * h:
        start_product_H[0] = 0
        H_zones = 1
        look_for_start = 0
    else:
        H_zones = 0
        look_for_start = 1

    # all lines
    for j in range(1,h):
        if look_for_start == 1:
            if rawsum_bm[j] > thresold * h:
                start_product_H[H_zones] = j
                H_zones += 1
                look_for_start = 0
        else:
            if rawsum_bm[j] < thresold * h and j-start_product_H[H_zones-1] > empiric_min_size:
                end_product_H[H_zones-1] = j
                look_for_start = 1

    if look_for_start == 0:
        if h-start_product_H[H_zones-1] > empiric_min_size:
            end_product_H[H_zones-1] = h-1
        else:
            H_zones -= 1

    # end of get horizontal borders
    return (start_product_H, end_product_H, H_zones)

def get_group_of_products(mask,start_product_H, end_product_H, H_zones):
    h, w = (np.array(mask.shape[:2])).astype(np.int)
    thresold = 0.3
    #empiric_min_size = 100
    # Get vertical borders
    V_zones = np.zeros(H_zones)
    start_product_V = np.zeros([H_zones,50])
    end_product_V = np.zeros([H_zones,50])

    for zone in range(H_zones):
        H_size = (end_product_H[zone] - start_product_H[zone]).astype(np.int)
        # compute colsum of binary_mask
        colsum_bm = 0
        for j in range(start_product_H[zone].astype(np.int),end_product_H[zone].astype(np.int)):
            colsum_bm = colsum_bm + mask[j,:]

        ## ANALYZE COLSUM
        # first col is products?
        if colsum_bm[0] > thresold * H_size:
            start_product_V[zone][0] = 0
            V_zones[zone] = 1
            look_for_start = 0
        else:
            V_zones[zone] = 0
            look_for_start = 1

        # all cols
        for j in range(1,w):
            if look_for_start == 1:
                if colsum_bm[j] > thresold * H_size:
                    #print 'new_zone', j, zone
                    start_product_V[zone][V_zones[zone]] = j
                    V_zones[zone] += 1
                    look_for_start = 0
            else:
                if colsum_bm[j] < thresold * H_size:# and j-start_product_V[zone][V_zones[zone]-1] > empiric_min_size:
                    #print 'end_zone', j, zone
                    end_product_V[zone][V_zones[zone]-1] = j
                    look_for_start = 1

        if look_for_start == 0:
            #if w-start_product_V[zone][V_zones[zone]-1] > empiric_min_size:
            end_product_V[zone][V_zones[zone]-1] = w-1
            #else:
            #    V_zones[zone] -= 1

    return start_product_V, end_product_V, V_zones

def get_product_segmentation(mask,start_shelf,end_shelf,start_gop,end_gop,ts=0.05):
    # init borders
    borders = np.empty(50)
    thresold = ts

    # compute vertical sum
    colsum = 0
    for j in range(int(start_shelf),int(end_shelf)):
        colsum = colsum + mask[j,start_gop:end_gop]

    ## ANALYZE COLSUM
   # if colsum[0] != 0:
   #     start_border = 0
   #     look_for_start = 0
   #     nbr_borders = 1
   # else:
   #     look_for_start = 1
   #     nbr_borders = 0

    # let's consider start start_gop is always a border
    start_border = 0
    look_for_start = 0
    nbr_borders = 1

    shelf_size = end_shelf - start_shelf
    gop_size = end_gop - start_gop
    for j in range(int(gop_size)):
        if look_for_start == 1:
            if colsum[j] > thresold * shelf_size:
                start_border = j
                nbr_borders += 1
                look_for_start = 0
        else:
            if colsum[j] < thresold * shelf_size:
                # border pos is middle of border width
                borders[nbr_borders-1] = int((j+start_border)/2 + start_gop)
                look_for_start = 1

    if look_for_start == 0:
        borders[nbr_borders-1] = end_gop
    else:
        nbr_borders += 1
        borders[nbr_borders-1] = end_gop

    return (borders, nbr_borders)


def remove_outlier(borders, nbr_borders):
    outliers = 1
    empiric_min_size = 30

    while outliers != 0 and nbr_borders != 1:
        tmp_borders = np.zeros(50)
        tmp_nbr_borders = 0

        # compute mean of product widths
        outliers = 0
        product_width = 0
        for k in range(nbr_borders-1):
            product_width += borders[k+1]-borders[k]

        product_width /= (nbr_borders-1.0)
        size_min = product_width/3.0
        if size_min < empiric_min_size:
            size_min = empiric_min_size

        #print size_min,'sm'

        tmp_borders[0] = borders[0]
        for k in range(nbr_borders-1):
            if borders[k+1]-borders[k] > size_min:
                tmp_nbr_borders += 1
                tmp_borders[tmp_nbr_borders] = borders[k+1]
            else:
                tmp_borders[tmp_nbr_borders] = (tmp_borders[tmp_nbr_borders] + borders[k+1])/2.0
                outliers += 1
                #print 'outlier removed'

        borders = tmp_borders
        nbr_borders = tmp_nbr_borders+1
        #print outliers, 'outliers removed'

    return (borders, nbr_borders)
    
    
def segmentation(im, mask_loc, mask_seg, ts=None, save = None):
    h, w = (np.array(im.shape[:2])).astype(np.int)
    result = np.zeros([h,w])
   
    objects = dict()
    nbr_products = 0

    # Get every shelf
    start_product_H, end_product_H, H_zones = get_shelf_position(mask_loc)

    # Start/End pos of each group of product in each shelf
    start_product_V, end_product_V, V_zones = get_group_of_products(mask_loc,start_product_H, end_product_H, H_zones)

    # Now look for segmentation results
    for j in range(H_zones):
        start_shelf =  start_product_H[j]
        end_shelf = end_product_H[j]
        #print start_shelf,end_shelf

        for i in range((V_zones[j]).astype(np.int)):
            start_gop = start_product_V[j][i]
            end_gop =  end_product_V[j][i]
            # result[int(start_shelf):int(end_shelf),int(start_gop):int(end_gop)] = 1
            # perform segmentation
            borders, nbr_borders = get_product_segmentation(mask_seg,start_shelf,end_shelf,start_gop,end_gop,ts=ts)

            # remove outliers (could be more 'global')
            borders, nbr_borders = remove_outlier(borders,nbr_borders)

            # check if zone has products
            if nbr_borders != 1:
                # print nbr_borders
                color = 2
                for k in range(nbr_borders-1):
                    result[int(start_shelf):int(end_shelf),int(borders[k])+5:int(borders[k+1])-5] = color
                    color += 1
                    objects[nbr_products] = [int(start_shelf),int(end_shelf),int(borders[k]),int(borders[k+1]),0,0,j] #item 6 is shelf number
                    nbr_products += 1
                    if color > 3:
                        color = 2
                            
    # be verbose !                         
   # result[np.where(result != 0)] = 1
    if not save == None:
        fake = cv2.applyColorMap(result.astype(np.uint8)*25, cv2.COLORMAP_HSV)
        fake[np.where(result == 0)] = 0
        res = im * 0.6 + fake * 0.4
        cv2.imwrite(save + '_SEG.jpg',res.astype(np.uint8))
        
        center_map = show_centers(im,objects)
        fake = cv2.applyColorMap(center_map.astype(np.uint8)*25, cv2.COLORMAP_HSV)
        fake[np.where(center_map == 0)] = 0
        res = im * 0.6 + fake * 0.4
        #cv2.imwrite(save + '_TARGET.jpg',res.astype(np.uint8))    
    
    return (result, objects, nbr_products)
 
   
def validate_objects(objects,compress=2.5):    
    # Search for objects of consistant and big size
    out = dict()
      
    size_w = np.zeros(len(objects))  
    size_h = np.zeros(len(objects))  
    for i in objects:
        pinfo = objects[i]
        w = pinfo[3] - pinfo[2]
        h = pinfo[1] - pinfo[0]
        size_w[i] = w        
        size_h[i] = h

    size_w = np.sort(size_w)    
    size_h = np.sort(size_h)
    
    # take eight of the biggest sizes to compute a max size
    max_w = np.mean(size_w[-10:-2])
    max_h = np.mean(size_h[-10:-2])
    # print max_w, max_h
    
    # valid objects must not differ too much from the max size
    out_valid = 0
    for i in objects:
        pinfo = objects[i]
        w = pinfo[3] - pinfo[2]       
        if (h > max_h/3) and (w > max_w/compress):
            out[out_valid] = pinfo
            out_valid += 1
            
    return out

# sums masks of all layers
def combine_layer_masks(mask):
    layers = len(mask)
    contribs = 1.0/layers
    h, w = np.shape(mask[0])
    out_mask = np.copy(mask[0])*contribs
    for layer in range(1,layers):
        out_mask += cv2.resize(mask[layer],(w, h))*contribs
              
    return out_mask
 
# resize a layer dictionary of objects to original size   
def resize_objects(objects, ratio):
    out = dict() 
    factor = 1.0/ratio
    for o in objects:
        pinfo = objects[o]
        for i in range(4):
            pinfo[i] = pinfo[i]*factor
        out[o] = pinfo
        
    return out


def find_credible_sizes(sizes):
    # size is a list of sizes (int)
    # let's look at how many close similar each layer has
    similar = np.zeros(len(sizes))
    max_sim = 0
    lay_max_sim = []
    nbr_layers = len(sizes)
    
    count = 0
    while max_sim == 0:     
        delta = 2+2*count
        # print delta, max_sim   
        for i in range(nbr_layers):
            my_size = sizes[i]
            for j in range(nbr_layers):
                if (j!=i) and (sizes[j] > my_size-delta) and (sizes[j] < my_size+delta):
                    similar[i] += 1
            if similar[i] > max_sim:
                max_sim = similar[i]
                lay_max_sim = [i]
            elif similar[i] == max_sim:
                lay_max_sim = lay_max_sim + [i]
        count += 1
                
    # best layer is lay_max_sim
    if len(lay_max_sim) == 1:
        best_layer = lay_max_sim[0]
    else:
        best_layer = 0
        best_size = 0
        for i in range(len(lay_max_sim)):
            if sizes[lay_max_sim[i]] > best_size:
                best_size = sizes[lay_max_sim[i]]
                best_layer = lay_max_sim[i]
      
    # estimate validity of a layer          
    for i in range(nbr_layers):
        if (sizes[i] > sizes[best_layer] - delta) and (sizes[i] < sizes[best_layer] + delta):
            similar[i] = 1
        else:
            similar[i] = 0              
            
    return similar
    

def registration(objects):
    nbr_layers = len(objects)    
    layer_sizes = []
    
    # let's select interesting layers
    if nbr_layers > 1:
        layer_sizes = []
        for layer in range(nbr_layers):
            layer_sizes = layer_sizes + [len(objects[layer])]
        good_layers = find_credible_sizes(layer_sizes)    
    else:
        good_layers = np.ones(1)    
    #print good_layers
    
    out = dict()
    correspondant = dict()
    tmp_dict = dict()      
    
    for layer in range(nbr_layers):
        if good_layers[layer] == 1:
            
            if len(tmp_dict) == 0:
                tmp_dict = objects[layer]
            else:
                for i in objects[layer]:
                    pinfo = objects[layer][i]
                    min_dist = 10000000
                    ref = -1
                    # find equivalent:
                    for o in tmp_dict:
                        obj = tmp_dict[o]
                        dist = compare_boxes_quadratic_distance(pinfo,obj)
                        if dist < min_dist:
                            min_dist = dist
                            ref = o
                    
                    
                    dist1, dist2 = compare_boxes(tmp_dict[ref],objects[layer][i])
                    # if box exist, save it
                    if dist1>0.5 and dist2>0.5:
                        if correspondant.has_key(ref):
                            correspondant[ref] += [layer, i]
                        else:
                            correspondant[ref] = [layer, i]
                    
                    # if product doesn't exist, fill it in tmp_dict 
                    else:
                        #print dist1, dist2
                        tmp_dict[len(tmp_dict)] = objects[layer][i]
     
     
    max_nbr_products = len(tmp_dict)
    max_correspondance = 1
    for obj in range(max_nbr_products):
        out_brands = dict()
        # si correspond à rien, pas pris en compte
        if correspondant.has_key(obj):
            ref_pinfo = tmp_dict[obj]
            """
            # get brands
            for j in range(5):
                out_brands[ref_pinfo[4][j]] = ref_pinfo[5][j]
            """
            nbr_correspondant = len(correspondant[obj])/2
            if nbr_correspondant + 1 > max_correspondance:
                max_correspondance = nbr_correspondant + 1
                
            for alt in range(nbr_correspondant):
                alt_pinfo = objects[correspondant[obj][alt*2]][correspondant[obj][alt*2+1]]
                
                # combine position
                for j in range(4):            
                    ref_pinfo[j] += alt_pinfo[j]
                """
                # combine brands
                for j in range(5):
                    if out_brands.has_key(alt_pinfo[4][j]):
                        out_brands[alt_pinfo[4][j]] += alt_pinfo[5][j]
                    else:
                        out_brands[alt_pinfo[4][j]] = alt_pinfo[5][j]
                """        
            # average position            
            for j in range(4):
                ref_pinfo[j] = int(ref_pinfo[j] / (nbr_correspondant+1))
            """    
            # average brands
            list_brands = np.zeros(len(out_brands))
            i = 0
            for j in out_brands:
                list_brands[i] = out_brands[j] / ( nbr_correspondant + 1 )
                i += 1
                
            idx, proba = get_five_best(list_brands)
            for j in range(5):
                ref_pinfo[4][j] = idx[j]
                ref_pinfo[5][j] = proba[j]                
            """    
            out[obj] = ref_pinfo + [(1.0+nbr_correspondant)/max_correspondance]               

        else:
            out[obj] = tmp_dict[obj] + [1.0/max_correspondance]           
 
    return out
  
    
def Image_seg(net_loc_and_seg, test_img, info_test, params, segmentation_type = 'dense'):  
    # Parameters:    
    Nw = params['segmentation_net_width']
    Nh = params['segmentation_net_height']
    overlap = params['overlap']   
    if params.has_key('pyramid_search'):
        pyramid = set_pyramid(params['pyramid_search'])
    else:
        pyramid = set_pyramid(0)

    # read input image
    #img = cv2.imread(image_test)
    img = np.array(test_img)
    img = img[:, :, ::-1].copy()
    #print np.shape(test_img)
    #cv2.imshow('img',test_img.astype('uint8'))
    ###cv2.waitKey(0)
	
    resize=1
    h, w = (np.array(img.shape[:2])).astype(np.int)
    while h < params['segmentation_net_height'] or w < params['segmentation_net_width']:
        img = cv2.resize(img,(w*2,h*2))
        h, w = (np.array(img.shape[:2])).astype(np.int)
        resize = 2*resize

    Lmask, Smask, imgs = Im_pyramidal_net_scan(img,net_loc_and_seg,Nw,Nh,overlap,pyramid)
    #cv2.imshow('test',imgs[0])
    #cv2.waitKey(0)
    # do GT loading
    if info_test != 0:
        GT_objects = dict()
        GT_mask = load_data_xls(img, info_test, GT_objects)
        print (len(GT_objects)), 'GT'
 
    # do global segmentation
    objects = dict()

    for layer in range(len(imgs)):
        if segmentation_type == 'dense':
            res, objects[layer], nbproducts = segmentation(imgs[layer],Lmask[layer],Smask[layer],ts=0.05)#,save = image_test[:-4]+ '_l' + str(layer))
            objects[layer] = validate_objects(objects[layer],compress=3) 
            #print objects[layer]
            objects[layer] = resize_objects(objects[layer],pyramid[layer])
        elif segmentation_type == 'sparse':
            #res, objects[layer], nbproducts = segmentation(imgs[layer],Lmask[layer],Smask[layer],ts=0.5)#,save = image_test[:-4]+ '_l' + str(layer))
            #objects[layer] = validate_objects(objects[layer],compress=3) 
            #objects[layer] = resize_objects(objects[layer],pyramid[layer])
            #cv2.imwrite('testL.jpg',Lmask[layer]*200)
            #cv2.imwrite('testS.jpg',Smask[layer]*200)

            objects[layer] = create_boxes(Lmask[layer])
            #show_boxes(imgs[layer],objects[layer],save='test')
            objects[layer] = clean_boxlist(objects[layer],Smask[layer],area_min=100) 
        
            #show_boxes(imgs[layer],objects[layer],save='test2')
        else:
            print 'ERROR: Wrong segmentation type'
            quit(0)

    objects = registration(objects) 
    #show_boxes(imgs[layer],objects,save='test3')
	
    for box in objects:
        for i in range(4):
            objects[box][i] = int(objects[box][i]/resize)
			
    return objects

def find_box(result,y,x,w,h):
    xmin = x
    xmax = x
    ymin = y
    ymax = y
    precision = 1
    
    pos_analyzed = np.zeros((w+2, h+2))
    set_to_analyze = [[x,y]]
    pos_analyzed[x,y] = 1
    iteration = 0
    
    while len(set_to_analyze) != 0:
        nb_points_to_test = len(set_to_analyze)
        x_to_analyze,y_to_analyze = set_to_analyze[nb_points_to_test-1]        
        del(set_to_analyze[nb_points_to_test-1])
        #if iteration%100==1: print iteration, len(set_to_analyze)
        
        if y_to_analyze + precision < h:
            if result[y_to_analyze+precision][x_to_analyze] != 0 and (pos_analyzed[x_to_analyze,y_to_analyze+precision]==0):
                set_to_analyze = set_to_analyze + [[x_to_analyze,y_to_analyze+precision]]
                pos_analyzed[x_to_analyze,y_to_analyze+precision] = 1
                if y_to_analyze+precision > ymax:
                    ymax = y_to_analyze+precision
        
        if x_to_analyze + precision < w:                
            if result[y_to_analyze][x_to_analyze+precision] != 0 and (pos_analyzed[x_to_analyze+precision,y_to_analyze]==0):
                set_to_analyze = set_to_analyze + [[x_to_analyze+precision,y_to_analyze]]
                pos_analyzed[x_to_analyze+precision,y_to_analyze] = 1
                if x_to_analyze+precision > xmax:
                    xmax = x_to_analyze+precision

        if y_to_analyze - precision > 0:                
            if result[y_to_analyze-precision][x_to_analyze] != 0 and (pos_analyzed[x_to_analyze,y_to_analyze-precision]==0):
                set_to_analyze = set_to_analyze + [[x_to_analyze,y_to_analyze-precision]]
                pos_analyzed[x_to_analyze,y_to_analyze-precision] = 1
                if y_to_analyze-precision < ymin:
                    ymin = y_to_analyze-precision

        if x_to_analyze - precision > 0:                
            if result[y_to_analyze][x_to_analyze-precision] != 0 and (pos_analyzed[x_to_analyze-precision,y_to_analyze]==0):
                set_to_analyze = set_to_analyze + [[x_to_analyze-precision,y_to_analyze]]
                pos_analyzed[x_to_analyze-precision,y_to_analyze] = 1
                if x_to_analyze-precision < xmin:
                    xmin = x_to_analyze-precision
        
        iteration += 1        
        #print set_to_analyze, pos_analyzed, xmin, xmax, ymin, ymax 
  
    return [xmin, xmax, ymin, ymax]           

def create_boxes(result):  
    nb_boxes = 0
    box_coords = dict()     # key: box_nbr, value: [xmin, xmax, ymin, ymax]
    
    height,width = result.shape[:2]
    ratio = 10
    result = cv2.resize(result,(width/ratio,height/ratio),0,0,cv2.INTER_NEAREST)
    height, width = result.shape[:2]
    
    for x in range(width):
        for y in range(height):
            
            if result[y][x] != 0:
                is_in_a_box = 0
                
                # bolongs to an existing box ?
                for box in box_coords:
                    cbox = box_coords[box]
                    if x >= cbox[0] and x <= cbox[1] and y >= cbox[2] and y <= cbox[3]:
                        is_in_a_box = 1
                    
                # create new box
                if not is_in_a_box:
                    box_coords[nb_boxes] = find_box(result,y,x,width,height)
                    #print box_coords
                    nb_boxes += 1
    # resize boxes
    for i in box_coords:
        box_coords[i] = [box_coords[i][0]*ratio,box_coords[i][1]*ratio,box_coords[i][2]*ratio,box_coords[i][3]*ratio]
        
    return box_coords
    
def show_boxes(im,boxlist,save='test'):
    for box in boxlist:
        cbox = boxlist[box]  
        dw = 10
        cv2.rectangle(im,(cbox[0],cbox[2]),(cbox[1],cbox[3]),(255,0,0),dw)
        
    cv2.imwrite(save+'_result.jpg',im)
    print 'done'
 
def segmask_cut(boxlist,segmask):
    outlist = dict()
    iterbox = 0
    for box in boxlist:
        cbox = boxlist[box]
        
        if cbox[1] > cbox[0] and cbox[3] > cbox[2]:
            colsum = 0
            for j in range(int(cbox[2]),int(cbox[3])):
                colsum = colsum + segmask[j,cbox[0]:cbox[1]]   
        
            look_for_start = 1 
            borders = [cbox[0]]
            thresold = 0.5
            
            Hsize = cbox[1] - cbox[0]
            Vsize = cbox[3] - cbox[2]
            for j in range(int(Hsize)):
                if look_for_start:
                    if colsum[j] > thresold * Vsize:
                        start_border = j
                        look_for_start = 0
                else:
                    if colsum[j] < thresold * Vsize:
                        # border pos is middle of border width
                        borders = borders + [int((j+start_border)/2 + cbox[0])]
                        look_for_start = 1  
            
            if look_for_start: # last is a border
                borders = borders + [cbox[1]]
             
            borders = remove_outlier(borders,len(borders))
            for i in range(len(borders)-1):
                outlist[iterbox] = [borders[i],borders[i+1],cbox[2],cbox[3]]
                iterbox += 1
            
    return outlist

            
def clean_boxlist(boxlist,segmask,area_min=10000):
    nb_boxes = len(boxlist)

    print '"""'
    print 'Boxes before clean:', nb_boxes
    
    seuil_perso = area_min  
    boxes_to_remove = []
    box_area_list = []    
    
    # cut with segmask
    ### ANTONIN ###
    #boxlist = segmask_cut(boxlist,segmask)    
    
    # calculer aire moyenne
    for box in boxlist:
        cbox = boxlist[box]

        c_area = (cbox[1]-cbox[0])*(cbox[3]-cbox[2])
        if c_area < seuil_perso:
            print 'Box removed with area', c_area
            boxes_to_remove = boxes_to_remove + [box]
        else:
            box_area_list = box_area_list + [c_area]
        
    for box in boxes_to_remove:
        del(boxlist[box])

    '''
    box_area_list = sorted(box_area_list)  
    boxes_to_remove = []
    to_remove_min = int(0.1*len(box_area_list))
    to_remove_max = len(box_area_list)-int(0.1*len(box_area_list))
    minimal_size = np.mean(box_area_list[to_remove_min:to_remove_max])/3.0
    maximal_size = np.mean(box_area_list[to_remove_min:to_remove_max])*3.0
    
    for box in boxlist:
        cbox = boxlist[box]
        c_area = (cbox[1]-cbox[0])*(cbox[3]-cbox[2])   
        if (c_area < minimal_size) or (c_area > maximal_size):
            print 'Box removed with area', c_area
            boxes_to_remove = boxes_to_remove + [box]
         
    for box in boxes_to_remove:
        del(boxlist[box])        
    '''

    out_dict = {}
    count = 0
    for box in boxlist:
        out_dict[count] = [boxlist[box][2], boxlist[box][3], boxlist[box][0], boxlist[box][1], 1]
        count += 1
        
    print 'Boxes after clean:', len(out_dict)
    return out_dict 


# function to crop vertically all images that have been segmented horizontally
def crop_for_text(number_crop, horizontal_bounds, reference_data,ref_segmentation):

    ## INPUT ###
    # number_crop is the number of crop we want in X dimension. The crops are applied on shelf images
    # reference_data is a dictionary containing all data : key : initial filename ; value : loaded image
    ## Horizontal_bound contains all vertical values of top shelf position

    # OUTPUT##
    # ref crop is a dictionary :
    #key : croped image name
    # values =[initial image name, Y start, X start ]of croped image

    # Find vertical position of shelves where we can crop without dividing a box

    factor=1
    ref_crop={}
    reference_data_croped={}
   
    for image_name in reference_data :

        im_ref =reference_data[image_name]
        w, h = im_ref.size

        bounds=[0] +horizontal_bounds[image_name]

        for u in range(0,len(bounds)):

            if u==len(bounds)-1:
                end_Y=h
            else:
                end_Y=int(bounds[u+1])
            if int(bounds[u])>100 : 
                startY=int(bounds[u])-100 # get larger crop
            else : 
                startY=0
      
            for i in range(0,number_crop):
            
                crop_X_low=i*int(w/number_crop)
                
                if i==number_crop-1:
                   crop_X_high=w
                else :
                   crop_X_high=(i+1)*int(w/number_crop)

                if 0<=crop_X_low< crop_X_high<=w and  0<=startY < end_Y<=h :
                
                    im=im_ref.crop((crop_X_low,startY,crop_X_high,end_Y))

                    reference_data_croped[str(image_name)+'_shelf_%s_crop_%s' %(u,str(i))]=im.resize((im.size[0] * factor, im.size[1] * factor))
                    ref_crop[str(image_name)+'_shelf_%s_crop_%s' %(u,str(i))]=[image_name,startY,crop_X_low]

    return  [ref_crop,reference_data_croped]

## remove outliers from segmentation boxes
def remove_outliers_seg(ref_segmentation):
    # ref_segmentation = {image_origin_name: {idbox: [y1, y2, x1, x2, index_of_shelf]}}

    ref_segmentation_out={}
    ## remove too small boxes
    for im in ref_segmentation : 
    
        boxes_to_remove = []   
        boxlist=ref_segmentation[im]
        
        #######    Remove outliers boxes    ###############
        # If larger < 20 pixel, hauteur <20 pixels or aire < 400 pixels then remove
        
        for box in boxlist:
            cbox = boxlist[box]
            c_area = (cbox[1]-cbox[0])*(cbox[3]-cbox[2])
            if c_area<200 or (cbox[1]-cbox[0])<20 or (cbox[3]-cbox[2])<20: #c_area < 6*mean_price_area:
                boxes_to_remove = boxes_to_remove + [box]
        if len(boxlist)>0 :
            print("Number of boxes removed is %i (%.2f percent)" % (len(boxes_to_remove), len(boxes_to_remove)/float(len(boxlist))))
        
        for box in boxes_to_remove:
            del(boxlist[box])

        ref_segmentation[im]=boxlist
    
        # Re_index ref_segmentation
        ref_segmentation_out[im]={}
        id_b=0
        for box_t in ref_segmentation[im] :
            ref_segmentation_out[im][id_b]=ref_segmentation[im][box_t]
            id_b=id_b+1
        
    return ref_segmentation_out
    
def qopius_segmentation(path_to_images,section,use_data=None,segmentation_type='dense'):

    # which domain?
    # set up networks

    if use_data == None:    
        net_root = shared_pathes.net_root_path
    else:
        net_root = use_data + '/saved_models/'
      
    segmentation_network = net_root + section + '/segmentation.prototxt'
    segmentation_weights = net_root + section + '/segmentation.caffemodel'
    net_fc_loc_and_seg = caffe.Net(segmentation_network, segmentation_weights, caffe.TEST)

    params = dict()
    params['segmentation_net_width'] = 512
    params['segmentation_net_height'] = 512
    params['overlap'] = 64 # This param can be adjusted    
    
    output = dict()
    
    for imgfile in path_to_images:
        image_test = path_to_images[imgfile]
        info_test = 0 # GT set to 0
        
        output[imgfile] = Image_seg(net_fc_loc_and_seg,image_test,info_test,params,segmentation_type=segmentation_type)
		
    return output

