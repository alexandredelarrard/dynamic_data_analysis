# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 15:00:39 2016

@author: ben

Utils functions for image and networks manipulation

Implements:

 - test_fullconv(fcnet, im)
     basic forward function for a given image through network fcnet
 - Im_pyramidal_net_scan(img,net_loc_and_seg,Nw,Nh,overlap,pyramid)
     performs pyramidal scan of img given ratios in pyramid
     returns Lmask, Smask, imgs (each ratio independently)
 - set_pyramid(p_search)
     constructs a default pyramid with p_search layers + 1
 - add_and_average(to_add,prev,count)
     Intelligent sum for overlapping scan
 
 - analyze_products(img,net,objects,Net_width,Net_height)
     Main forward function for product IDENTIFICATION
 - get_five_best(score)
     generates a top five from the network output
 
 - combine_masks_products_found(img,GT,mask)
     can be used to display mask of 'correct identification'
 
"""
import numpy as np
import cv2
from images_and_masks import correct_picture_ratio

def add_and_average(to_add,prev,count):
  
    tmp = np.multiply(prev,count-1)
    tmp += to_add
    tmp = np.divide(tmp,count)
    
    return tmp

def test_fullconv(fcnet, im):
    h = np.int(im.shape[0])
    w = np.int(im.shape[1])
    fcnet.blobs['data'].reshape(1, 3, h, w)
    mean = np.array([100, 109, 113])
    mean = mean[:, np.newaxis, np.newaxis]
    dd = np.zeros((1, 3, h, w))
    dd[0] = np.rollaxis(im, 2, 0)
    dd -= mean
    out = fcnet.forward(data=dd)
    return out
    
def set_pyramid(p_search):
    p_search += 1
    pyramid = np.zeros([p_search])
    for p in range(p_search):
        if p==0:
            pyramid[p] = 1
        elif p%2==1:
            pyramid[p] = 1/(1+0.2*int((p+1)/2))
        else:
            pyramid[p] = 1+0.2*int((p+1)/2)

    pyramid = np.sort(pyramid)
        
    return pyramid
    
def Im_pyramidal_net_scan(img,net_loc_and_seg,Nw,Nh,overlap,pyramid):
    Lmask = dict()
    Smask = dict()
    imgs = dict()
    h, w = (np.array(img.shape[:2])).astype(np.int)
  
    if len(pyramid) != 1:
        print 'Pyramid search selected,', len(pyramid), 'layers used'
    
    iteration = 0
    for ratio in pyramid:
        posy = 0
        end_y = 0
        nw = int(w*ratio)
        nh = int(h*ratio)
        img_test = cv2.resize(img,(nw,nh))
        tmp_Lmask = np.zeros([nh,nw])      
        tmp_Smask = np.zeros([nh,nw])
        contribs = np.zeros([nh,nw])
        #print 'Psearch -', ratio
        
        while posy < nh:
            posx = 0
            end_x = 0
            
            while posx < nw:

                #print 'Scanning position:',posy,posy+Nh
                im = img_test[posy:posy+Nh,posx:posx+Nw]
      
                # compute local mask
                mask_loc = test_fullconv(net_loc_and_seg, im.copy())  # ['prob'][0]
                mask_loc = mask_loc['prob'][0].argmax(axis=0)
                mask_seg = np.copy(mask_loc)
    
                mask_loc[np.where(mask_loc != 0)] = 1 
                mask_seg[np.where(mask_seg != 2)] = 0
                mask_seg[np.where(mask_seg == 2)] = 1      
                #mask_seg = mask_seg.argmax(axis=0)  

                # average with previous
                contribs[posy:posy+Nh,posx:posx+Nw] += 1
                tmp_Lmask[posy:posy+Nh,posx:posx+Nw] = add_and_average(mask_loc,tmp_Lmask[posy:posy+Nh,posx:posx+Nw],contribs[posy:posy+Nh,posx:posx+Nw])
                tmp_Smask[posy:posy+Nh,posx:posx+Nw] = add_and_average(mask_seg,tmp_Smask[posy:posy+Nh,posx:posx+Nw],contribs[posy:posy+Nh,posx:posx+Nw])
               
                # update posx
                posx += Nw - overlap
                if posx + Nw > nw-1:
                    
                    if end_x == 1:
                        posx = nw
                    else:
                        posx = nw-1-Nw
                        end_x = 1

            # update posy
            posy += Nh - overlap

            if posy + Nh > nh-1:

                if end_y == 1:

                    posy = nh
                else:
                    posy = nh-1-Nh

                    end_y = 1
                   
            
       

        # resize and update masks
        tmp_Lmask[np.where(tmp_Lmask > 0.5)] = 1
        tmp_Lmask[np.where(tmp_Lmask != 1)] = 0
        tmp_Smask[np.where(tmp_Smask != 0)] = 1
        tmp_Smask[np.where(tmp_Smask != 1)] = 0        
        
        Lmask[iteration] = tmp_Lmask
        Smask[iteration] = tmp_Smask    
        imgs[iteration] = img_test
        iteration += 1

        
        #Lmask += cv2.resize(tmp_Lmask,(w,h))
        #Smask += cv2.resize(tmp_Smask,(w,h))

    return Lmask, Smask, imgs
    """
    Lmask[np.where(Lmask > 0.5)] = 1
    Lmask[np.where(Lmask != 1)] = 0
    Smask[np.where(Smask != 0)] = 1
    Smask[np.where(Smask != 1)] = 0

    cv2.imwrite('output/tmp/Lmask.JPG',Lmask.astype(np.uint8)*255)
    cv2.imwrite('output/tmp/Smask.JPG',Smask.astype(np.uint8)*255)
    """
 
def Im_net_scan(img,net_loc_and_seg,Nw,Nh,overlap,Lmask=None,Smask=None,imgs=None,ratio=1,iteration=0):
    if Lmask == None:    
        Lmask = dict()
    if Smask == None:    
        Smask = dict()
    if imgs == None:    
        imgs = dict()
    h, w = (np.array(img.shape[:2])).astype(np.int)
  
    posy = 0
    end_y = 0
    nw = int(w*ratio)
    nh = int(h*ratio)
    img_test = cv2.resize(img,(nw,nh))
    tmp_Lmask = np.zeros([nh,nw])      
    tmp_Smask = np.zeros([nh,nw])
    contribs = np.zeros([nh,nw])
    
    while posy < nh:
        posx = 0
        end_x = 0
        
        while posx < nw:
            #print 'Scanning position:',posy,posy+Nh
            im = img_test[posy:posy+Nh,posx:posx+Nw]
  
            # compute local mask
            mask_loc = test_fullconv(net_loc_and_seg, im.copy())  # ['prob'][0]
            mask_loc = mask_loc['prob'][0].argmax(axis=0)
            mask_seg = np.copy(mask_loc)

            mask_loc[np.where(mask_loc != 0)] = 1 
            mask_seg[np.where(mask_seg != 2)] = 0
            mask_seg[np.where(mask_seg == 2)] = 1      
            #mask_seg = mask_seg.argmax(axis=0)            

            # average with previous
            contribs[posy:posy+Nh,posx:posx+Nw] += 1
            tmp_Lmask[posy:posy+Nh,posx:posx+Nw] = add_and_average(mask_loc,tmp_Lmask[posy:posy+Nh,posx:posx+Nw],contribs[posy:posy+Nh,posx:posx+Nw])
            tmp_Smask[posy:posy+Nh,posx:posx+Nw] = add_and_average(mask_seg,tmp_Smask[posy:posy+Nh,posx:posx+Nw],contribs[posy:posy+Nh,posx:posx+Nw])

            # update posx
            posx += Nw - overlap
            if posx + Nw > nw-1:
                
                if end_x == 1:
                    posx = nw
                else:
                    posx = nw-1-Nw
                    end_x = 1
            

        # update posy
        posy += Nh - overlap
        if posy + Nh > nh-1:
            if end_y == 1:
                posy = nh
            else:
                posy = nh-1-Nh
                end_y = 1
                
    # resize and update masks
    tmp_Lmask[np.where(tmp_Lmask > 0.5)] = 1
    tmp_Lmask[np.where(tmp_Lmask != 1)] = 0
    tmp_Smask[np.where(tmp_Smask != 0)] = 1
    tmp_Smask[np.where(tmp_Smask != 1)] = 0        
    
    Lmask[iteration] = tmp_Lmask
    Smask[iteration] = tmp_Smask    
    imgs[iteration] = img_test
    return Lmask, Smask, imgs
   
def get_five_best(score):
    found = np.zeros(5)
    proba = np.zeros(5)    
    for i in range(5):
        found[i] = score.argmax(axis=0)                     
        proba[i] = score[found[i]]        
        score[found[i]] = 0
        
    return found, proba
 
 
def analyze_products(img,net,objects,Net_width,Net_height):
    image_height, image_width = img.shape[:2]
    proportion_x = 0.5
    proportion_y = 0.5    
    mask = np.zeros([image_height,image_width])
    
    # set transformer for data manipulation
    desired_ratio = float(Net_width) / float(Net_height) 
    for i in objects:
        pinfo = objects[i]
        w = pinfo[3] - pinfo[2]
        h = pinfo[1] - pinfo[0]
        dw = w * (1 - proportion_x)/2
        dh = h * (1 - proportion_y)/2
        
        x1 = int(max(0,pinfo[2]-dw))            
        x2 = int(min(image_width-1, pinfo[3]+dw))
        y1 = int(max(0,pinfo[0]-dh))
        y2 = int(min(image_height-1, pinfo[1]+dh))            
            
        product_image = img[y1:y2,x1:x2]
        # We have found our products                        
        pimg = correct_picture_ratio(product_image, desired_ratio)
        outimg = cv2.resize(pimg, (Net_width, Net_height))
       # cv2.imshow('img',outimg.astype(np.uint8))        
       # cv2.waitKey(0)
        
        # go through net
        score = test_fullconv(net, outimg.copy())  # ['prob'][0]
        score = score['prob'][0]   
        
        # get best scores
        found, proba = get_five_best(score)  
        mask[y1:y2,x1:x2] = found[0]                
        pinfo[4] = found
        pinfo[5] = proba
    return mask
    
     
def combine_masks_products_found(img,GT,mask):
    h, w = np.array(img.shape[:2]).astype(np.int)
    # reinit all labels to 1
    GT[np.where(GT != 0)] = 1

    mask_primary = np.zeros([h,w])
    mask_primary[np.where(mask == 1)] = 1

    mask_secondary = np.zeros([h,w])
    mask_secondary[np.where(mask == 2)] = 1    

    mask[np.where(mask != 0)] = 1
 
    # compute stats
    union = mask + GT

    # get union 
    union[np.where(union != 0)] = 1    
    GT_only = union - mask
    GT_only[np.where(GT_only != 1)] = 0
    
    # save product targets
    visual_result = np.zeros([h,w,3])
       
    # set colors
    visual_result[:,:,0] = mask_secondary*255
    visual_result[:,:,1] = mask_primary*255
    visual_result[:,:,2] = GT_only*255    
        
    # be verbose
    visual_result = img * 0.6 + visual_result * 0.4
    cv2.imwrite('output/visualisations/Product_identification.jpg',visual_result.astype(np.uint8))    
        
    return visual_result