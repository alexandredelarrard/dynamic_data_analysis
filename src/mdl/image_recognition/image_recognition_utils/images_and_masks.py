# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 12:58:22 2016

@author: ben

Utils functions for image and mask manipulation

Implements:

 - correct_picture_ratio(I, desired_ratio, mode=cv2.BORDER_CONSTANT)
     used to resize image while preserving ratio
 - show_objects(img,objects, save = None)
     displays an dictionary of objects on image, saving in save.
 - show_centers(im,obj,r = 17,value = 1)
     displays centers of objects in an image
 - show_mask(img, mask, scale=1.0, color_scaling=20)
     displays mask over the image
 - mask_resizer(mask,w,h)
      resizes mask to given size and preserves classes   
 - combine_masks(img,GT,mask, save = 'output/visualisations/Compared_segmentation.jpg')     
       generates a mask showing the similarity between two given masks   
 
"""
import cv2
import numpy as np
from fct_getters import get_brand_name_from_brand


def show_objects(img, objects, show = None, save = None, chroma = None, price_box=None):
    image_height, image_width = img.shape[:2]
    mask = np.zeros([image_height, image_width])    

    for o in objects:
        pinfo = objects[o]
        if show == None:
            mask[pinfo[0]:pinfo[1],pinfo[2]+10:pinfo[3]-10] = 1
        elif show == 'info_7':
            #if pinfo[7]>0.5:
            mask[pinfo[0]:pinfo[1],pinfo[2]+10:pinfo[3]-10] = int(pinfo[7]*10)
        elif show == 'info_4':
            mask[pinfo[0]:pinfo[1],pinfo[2]+10:pinfo[3]-10] = (int(pinfo[4][0])+1)%255
        elif show == 'info_5':
            mask[pinfo[0]:pinfo[1],pinfo[2]+10:pinfo[3]-10] = int((pinfo[5][0])*2)
        ## Kevin
        elif show == 'test_kevin':
            mask[pinfo[0]:pinfo[1],pinfo[2]+10:pinfo[3]-10] = chroma[int(pinfo[4][0])][0]
        ## Kevin

    if not save == None:    
        fake = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_HSV)
        fake[np.where(mask == 0)] = 0
        res = img * 0.5 + fake * 0.5

        ## Kevin
        if show == 'test_kevin':
            print "- printing brands on boxes...",
            font = cv2.FONT_HERSHEY_SIMPLEX
            for box in objects:
                # brand name
                height_box = objects[box][1] - objects[box][0]
                possible_nb_letters = int(height_box/40)  # 40 pixel by letter
                brand_id = int(objects[box][4][0])
                brand_name = chroma[brand_id][1]
                if brand_name:
                    truncated_brand_name = str(brand_name[0:possible_nb_letters])
                else:
                    truncated_brand_name = "*"
                for l in range(len(truncated_brand_name)):
                    x = int(objects[box][2])  # x1
                    y = int(objects[box][1]) - 40*(len(truncated_brand_name) - l)  # y2
                    org_l = (x, y)
                    cv2.putText(res, truncated_brand_name[l], org_l, font, 1.5, (255,255,255), 4)  # last three arguments are text size, color, boldness

                # confidence
                if objects[box][3] - objects[box][2] > 70 and truncated_brand_name != "*":
                    org_confi = (int(objects[box][3]-70), int(objects[box][1])) 
                    confi = str(objects[box][5][0])[0:4]
                    cv2.putText(res, confi, org_confi, font, 1, (255,255,255), 4)

                # box number
                if objects[box][3] - objects[box][2] > 45 and objects[box][1] - objects[box][0] > 35:
                    org_confi = (int(objects[box][3]-45), int(objects[box][0])+35) 
                    confi = str(box)
                    cv2.putText(res, confi, org_confi, font, 1, (255,255,255), 3)

                # price

                #try : 
                if not price_box == None :

                    if objects[box][3] - objects[box][2] > 30 and objects[box][1] - objects[box][0] > 30:
                        if box in price_box : 
                            org_price = (int((objects[box][2]+objects[box][3])/2-20), int((objects[box][0]+objects[box][1])/2)) 
                            price = str(price_box[box][4])+'$'
                            cv2.putText(res, price, org_price, font, 1, (0,0,255), 3)
                
                #except : 
                #       print 'price in wrong format cannot be printed'

        ## Kevin

        cv2.imwrite(save,res.astype(np.uint8))

    return mask
    
# correct pictures ratio to fit with network
# it adds black columns on the border to preserve the original aspect
def correct_picture_ratio(I, desired_ratio, mode=cv2.BORDER_CONSTANT):
    height, width = I.shape[:2]
    orig_ratio = float(width) / height
    border_img = I
    diff_ratio = abs(orig_ratio - desired_ratio) / (orig_ratio + desired_ratio)
    if diff_ratio > 0.05:
        # make h2
        if desired_ratio > orig_ratio:  # image is flatter
            h2 = height
            w2 = int(h2 * desired_ratio)
        else:
            w2 = width
            h2 = int(w2 / desired_ratio)

        dW = max(0, w2 - width)
        dH = max(0, h2 - height)
        dT = dB = dH / 2
        dL = dR = dW / 2
        border_img = cv2.copyMakeBorder(I, dT, dB, dL, dR, cv2.BORDER_CONSTANT)
    return border_img

def show_centers(im,obj,r = 17,value = 1):        
    h, w = (np.array(im.shape[:2])).astype(np.int)
    mask = np.zeros([h,w]) 
    for i in obj:
        c_obj = obj[i]
        cx = int((c_obj[2]+c_obj[3])/2)
        cy = int((c_obj[0]+c_obj[1])/2)        
        mask[cy-r:cy+r,cx-r:cx+r] = value   
                
    return mask 

# Shows a mask.
def show_mask(img, mask, iterr, scale=1.0, color_scaling=20):
    h, w = (np.array(img.shape[:2]) * scale).astype(np.int)
    im = cv2.resize(img, (w, h))
    gtR = cv2.resize(mask.astype(np.float32), (w, h))

    gtR = gtR * color_scaling

    fake = cv2.applyColorMap(gtR.astype(np.uint8), cv2.COLORMAP_HSV);
    fake[np.where(gtR == 0)] = 0
    res = im * 0.6 + fake * 0.4
    
    cv2.imwrite('res' +str(iterr)+'.jpg', res.astype(np.uint8))
    if iterr > 100:
        quit(0)
    #cv2.waitKey(0)

    return res

def combine_masks(img,GT,mask, save = 'output/visualisations/Compared_segmentation.jpg'):
    h, w = np.array(img.shape[:2]).astype(np.int)
    # reinit all labels to 1
    GT[np.where(GT != 0)] = 1
    mask[np.where(mask != 0)] = 1
    
    # compute stats
    union = mask + GT

    # get union and intersection
    inter = np.zeros([h,w]) 
    inter[np.where(union != 2)] = 0
    inter[np.where(union == 2)] = 1
    union[np.where(union != 0)] = 1    
    
    mask_only = union - GT
    mask_only[np.where(mask_only != 1)] = 0
    GT_only = union - mask
    GT_only[np.where(GT_only != 1)] = 0
    
    # save product targets
    visual_result = np.zeros([h,w,3])
    visual_result[:,:,1] = mask*255
       
    # set colors
    visual_result = visual_result * 0
    visual_result[:,:,0] = mask_only*255
    visual_result[:,:,1] = inter*255
    visual_result[:,:,2] = GT_only*255    
    
    #area_inter = sum(map(sum, inter))
    #area_GT = sum(map(sum, GT))
    
    # be verbose
    visual_result = img * 0.6 + visual_result * 0.4
    cv2.imwrite(save,visual_result.astype(np.uint8))            
    return visual_result
    
# "Smart" mask resizer (interpolation preserves classes)
def mask_resizer(mask,w,h):
    output=np.zeros((h,w))
    numc=int(np.max(mask))
    for c in range(numc):
        i = c+1
        tmp = np.zeros_like(mask)
        tmp[np.where(mask==i)]=255
        tmp2 = cv2.resize(tmp,(w,h),0,0,cv2.INTER_LINEAR)
        #clean interpolated value
        output[np.where(tmp2>254)]=i
    return output
