# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 13:02:18 2016

@author: ben

Utils for in/out data manipulation

Implements:

 - show_hdf5(filename)
     displays images stored in a image+mask database
 - split_hdf5(dbname, nbr_samples, nbr_files, tr=0.7)
     splits a given hdf5 file (dbname) and generates 
 
 - get_brand_per_label(label)
     takes the integer label and returns the corresponding brand
 - load_data_xls(image, boxfile, product_dico, load_borders = 0)
     reads information of a standard xls file
     TODO; add option for reading what is generated via save_objects_in_xls
 - save_objects_in_xls(objects, save = '/output/visualisations/', convert_scores = 0)
     save a dict in an xls using save + 'result.xls' name. Use convert_scores when your boxes contain top5 scores

 - create_imgs_from_objects(img,obj, save = 'output/products/')
     generate a img per object
     
 - combine_objects(GT_obj,obj,identify = 0)
     performs estimation of similarity between the Groud truth and the found object set
 - compare_boxes(i1,i2,motion = [0, 0])
     estimate similarity between two boxes. Motion compensation available 
 
"""
import numpy as np
import sys, os
import h5py
import cv2
import pyexcel as pe
from pyexcel_xls import save_data
from collections import OrderedDict
import pyexcel.ext.xls

import COMMON as shared_pathes
sys.path.append(shared_pathes.caffe_root+'distribute/python/')
#import caffe


from images_and_masks import show_mask
## THIS SHOULD BE MADE GLOBAL!!! USE IT AS A REFERENCE WHEN TRAINING
net_val_labels = {u'LOVEA': 153, u'REVLON': 77, u'BOURJOIS': 151, u'RUFFLES': 89, u'LIGHT BLUE SHADE': 37, u'AXE': 8, u"L'OREAL": 3, u'MUNCHIES': 102, u'NIVEA': 142, u'MARC ANTHONY': 25, u'EUGEN PERMA': 139, u'PROVOST': 138, u'PERT': 13, u'AUSSIE': 30, u'CHI': 45, u'SEXY HAIR': 48, u'WHITE RAIN': 118, u'BATISTE': 42, u'TIO NACHO': 107, u'LIVING PROOF': 124, u'PAUL MITCHELL': 115, u'PRESIDENT CHOICE': 67, u'LIVE CLEAN': 35, u'CHESTERS': 100, u'FEKKAI': 50, u'NEUTROGENA': 33, u'MARKET PANTRY': 131, u'KELLOGGS': 55, u'DORITOS': 91, u'GUD': 120, u'HERBAL ESSENCES': 10, u'MANE N TAIL': 32, u'DMP': 143, u'VO5': 44, u'ROLD GOLD': 96, u'SIGNATURE': 132, u'NIZORAL': 104, u'CO LABS': 136, u'ARGILE': 152, u'GOSH': 46, u'DOWN UNDER': 31, u'ALBA BOTANICA': 106, u'MATRIX': 19, u'BEANITOS': 86, u'SELSUN BLUE': 41, u'NATURE VALLEY': 59, u'KRETSCHMER': 65, u'CLAIROL': 76, u'LAYS': 90, u'BEAR NAKED': 134, u'TEATREE': 116, u'SUAVE': 113, u'ATTITUDE': 121, u'GARNIER': 11, u'ALGEMARIN': 137, u'IT': 112, u'RENPURE ORIGINALS': 105, u'JOHN FRIEDA': 24, u'CARNATION': 71, u'LA COUPE': 43, u'POST': 53, u'AMERICAN CREW': 128, u'FOURMI BIONIQUE': 74, u'SELECTION': 75, u'BIOSILK': 22, u'FRITO': 98, u'BED HEAD': 21, u'COLOURB4': 79, u'EQUATE': 2, u'MOM': 135, u'WEETABIX ALPEN': 60, u'2CHIC': 122, u'HOSTESS': 94, u'MUNCHOS': 101, u'UNKNOWN': 69, u'QUAKER': 56, u'KERASTASE': 126, u'ALBERTO': 14, u'RAVE': 117, u'COMPLIMENTS': 57, u'TRESEMME': 5, u'TOSTITOS': 99, u'GENERAL MILLS': 54, u'LE PETIT OLIVIER': 149, u'CHEETOS': 92, u'LA LOOKS': 108, u'PRELL': 130, u'BARBARA GOULD': 144, u'VIDAL SASSOON': 9, u'YUM YUM': 80, u'BEDHEAD': 51, u'MACADEMIA': 49, u'DIPPITY DO': 16, u'CORINE DE FARME': 150, u'PANTENE': 1, u'SPLAT': 78, u'SMARTFOOD': 93, u'PRESTIGE DE FRANCE': 147, u'HASK': 38, u'SUN CHIPS': 95, u'WEETABIX': 62, u'KARDASHIAN BEAUTY': 127, u'FAT HAIR': 111, u'DOVE': 4, u'ORGANIX': 26, u'BUGLES': 88, u'LOBLAWS': 81, u'TARGET': 119, u'CASCADIAN FARM': 133, u'NEXXUS': 20, u'INNOVATION': 40, u'CREAM OF WHEAT': 64, u'SG': 110, u'PRIOBIOTIC CEREAL': 73, u'KASHI': 70, u'DIADERMINE': 146, u'FINESSE': 15, u'DAILY': 17, u'ORVILLE': 82, u'OLD SPICE': 12, u'AQUA NET': 109, u'INFUSIUM 23': 28, u'LABORATOIRE COSMEPRO': 36, u'SWISSLI': 68, u'SEBASTIAN PROFESSIONAL': 47, u'JORDANS': 61, u'AVEENO': 27, u'MISS VICKIES': 97, u'MIXA': 148, u'ROBIN HOOD': 63, u'KLORANE': 39, u'CLEAR': 29, u'SALLYS': 66, u'OLD DUTCH': 85, u'POP CHIPS': 84, u'PUREOLOGY': 114, u'SYOSS': 141, u'JOICO': 18, u'HARVEST SNAPS': 87, u'OTHER': 7, u'FRENCH FORMULA': 52, u'DORISTOS': 103, u'TONY GUY': 123, u'NATURES PATH': 72, u'REDKEN': 23, u'SCHWARZKOPF': 140, u'AUCHAN': 145, u'HEAD & SHOULDERS': 6, u'PROBIOTIC CEREAL': 58, u'GOT2B': 34, u'SHEA MOISTURE': 129, u'CAPE COD': 83, u'DESSANGE': 125}

# For viewing a prepared hdf5
def show_hdf5(dbname):
    f = h5py.File(dbname, 'r')
    dbimg = f["Images"]
    dbmask = f["Mask"]
    print 'Image db shape : ', dbimg.shape
    print 'Mask db shape : ', dbmask.shape
    shape = (1, 3, dbimg.shape[2], dbimg.shape[3])

    transformer = caffe.io.Transformer({'data': shape})
    transformer.set_mean('data', np.array([100, 109, 113]))
    transformer.set_transpose('data', (2, 0, 1))

    for i in range(dbimg.shape[0]):
        im0 = dbimg[i]
        im = transformer.deprocess('data', im0)
        mask = dbmask[i]
        res = show_mask(im, mask, i)
    f.close()

# Separate H5PY into training/testing sets
# TODO: generate automatic logs of what has been done during the loading phase
def split_hdf5(dbname, nbr_samples, nbr_files, tr=0.7):
    f = h5py.File(dbname, 'r')
    dbimg = f["Images"]
    dbmask = f["Mask"]
     
    dir = os.path.dirname(dbname) + '/'
    img_nbr = 0

    n, c, h, w = dbimg.shape
    n_per_file = nbr_samples/nbr_files
    
    if dbmask.shape[1] == 1:
        load_labels = 1
    else:
        load_labels = 0
        n, h2, w2 = dbmask.shape

    for iter in range(nbr_files):
        #print iter, n_per_file
        NTrain = np.ceil(n_per_file * tr)
        NTest = n_per_file - NTrain
        ftrain = h5py.File(dir + 'train' + str(iter) + '.hd5', 'w')
        ftrain.create_dataset("Images", (NTrain, c, h, w), dtype='float32')
        if load_labels == 1:
            ftrain.create_dataset("Mask", (NTrain, 1), dtype='float32')        
        else:                
            ftrain.create_dataset("Mask", (NTrain, h2, w2), dtype='float32')
        
        ftest = h5py.File(dir + 'test' + str(iter) + '.hd5', 'w')
        ftest.create_dataset("Images", (NTest, c, h, w), dtype='float32')
        if load_labels == 1:
            ftest.create_dataset("Mask", (NTest, 1), dtype='float32')       
        else:
            ftest.create_dataset("Mask", (NTest, h2, w2), dtype='float32')
        
        idx = range(n_per_file);
        np.random.shuffle(idx);
        for i in idx:
            #print i
            if i < NTrain:
                ftrain["Images"][i] = dbimg[img_nbr]
                ftrain["Mask"][i] = dbmask[img_nbr]
            else:
                ftest["Images"][i - NTrain] = dbimg[img_nbr]
                ftest["Mask"][i - NTrain] = dbmask[img_nbr]
            img_nbr = img_nbr + 1
      
    print img_nbr, 'images have been written'
 

def create_imgs_from_objects(img,obj, save = 'output/products/'):
    margin = 20    
    it = 0
    h, w = (np.array(img.shape[:2])).astype(np.int)
    
    for i in obj:
        ob=obj[i]
        it += 1
        
        y1 = max(0,ob[0]-margin)        
        y2 = min(h-1,ob[1]+margin)        
        x1 = max(0,ob[2]-margin) 
        x2 = min(w-1,ob[3]+margin)
        
        Im_ob = img[y1:y2,x1:x2] 
        
        Im_name = save + 'Product' + str(it) + '.jpg'        
        cv2.imwrite(Im_name,Im_ob.astype(np.uint8))    
        
def combine_objects(GT_obj,obj,identify = 0):
    false_alarms = 0
    correct_identification = 0
    secondary_identification = 0
    found_product = np.zeros(len(GT_obj)) 
    found_unique = 0
    found_overlap = 0
    found_multiple = 0
    min_overlap = 0.3
    found_match = dict()
   
    # build dico of found objects
    for o in obj:
        # check if match is found in GT
        best_match = -1
        nbr_touched = 0
        max_overlap = 0
        for GTo in GT_obj:
            scores = compare_boxes(obj[o],GT_obj[GTo])
            if scores[0] > min_overlap:
                found_match[nbr_touched] = GTo
            if scores[0] > max_overlap:
                max_overlap = scores[0]
                best_match = GTo
                
        #print obj[o], found_match
        if len(found_match) == 0:
            false_alarms += 1
        else:
            if len(found_match) == 1:
                found_unique += 1
            else:
                found_overlap += 1
            for j in found_match:
                found_product[found_match[j]] += 1
                if(found_product[found_match[j]] > 1):
                    found_multiple += 1
                    
        # compare idenfiers
        if identify == 1:
            if best_match != -1:
                if  net_val_labels[GT_obj[best_match][4]]==int(1+obj[o][4][0]):
                    correct_identification += 1
                else:
                    for secondary_guess in range(1,5):
                        if  net_val_labels[GT_obj[best_match][4]]==int(1+obj[o][4][secondary_guess]):
                            secondary_identification += 1
                    
            #print 'GT', GT_obj[best_match][4], '( label', net_val_labels[GT_obj[best_match][4]],') - found', int(1+obj[o][4][0])
            
            
    found_once = len(found_product[np.where(found_product!=0)])
    
    print ''
    print '-- SEGMENTATION RESULTS --'
    print 'We found ', len(obj), 'boxes'
    print 'Products found at least once :', found_once, '/', len(GT_obj), '(GT) -', float(found_once)/len(GT_obj)*100, '% accuracy'
    print 'Products missed :', len(GT_obj)-found_once
    print 'False alarms :', false_alarms      
    print 'Products found multiple times :', found_multiple
    print 'Boxes that contain more than one product :', found_overlap   
    
    if identify == 1:    
        print ''
        print '-- IDENTIFICATION RESULTS --'
        print correct_identification, 'correct product recognition', float(correct_identification)/len(GT_obj)*100, '% accuracy'
        total = min(found_once,correct_identification + secondary_identification) # should not count multiple times the same...
        print total, 'in top 5', float(total)/len(GT_obj)*100, '% accuracy'
        
    stats = [found_unique, found_once]
    return stats
    
def compare_boxes(i1,i2,motion = [0, 0]):
    ay1 = i1[0]    
    ay2 = i1[1]
    ax1 = i1[2]
    ax2 = i1[3]
    by1 = i2[0]-motion[0]
    by2 = i2[1]-motion[0]
    bx1 = i2[2]-motion[1]
    bx2 = i2[3]-motion[1]
    shared_y = np.min([ay2,by2])-np.max([ay1,by1])        
    shared_x = np.min([ax2,bx2])-np.max([ax1,bx1])        
    
    if shared_y > 0 and shared_x > 0:
        #check similar sizes
        area2 = (by2-by1)*(bx2-bx1)        
        area1 = (ay2-ay1)*(ax2-ax1)        
        area_shared = shared_x * shared_y      
        overlap = area_shared*1.0/max([area2,area1])
        similarity = min([area2,area1])*1.0/max([area2,area1])
        return [overlap, similarity]
    else:
        return [0, 0]
        
def compare_boxes_quadratic_distance(i1,i2,motion = [0, 0]):
    ay1 = i1[0]    
    ay2 = i1[1]
    ax1 = i1[2]
    ax2 = i1[3]
    by1 = i2[0]-motion[0]
    by2 = i2[1]-motion[0]
    bx1 = i2[2]-motion[1]
    bx2 = i2[3]-motion[1]
    # compute mean quadratic distance
    MQD = (ay1-by1)*(ay1-by1)+(ay2-by2)*(ay2-by2)+(ax1-bx1)*(ax1-bx1)+(ax2-bx2)*(ax2-bx2)    
    
    return MQD    
    
# function to load brand info from xlsx file.
# It iteratively fills maxnames (current total number of brands) and lutnames (storing brand names)
# The mask is filled with the corresponding label for each box
def load_data_xls(image, boxfile, product_dico, load_borders = 0, section = None, get_brand = 0):
    # how to correctly load getters ?
    path_to_utils = os.path.dirname(os.path.realpath(__file__)) + '/../data/general_database_storage/utils/'
    sys.path.append(path_to_utils)
    import getters as gt
 
    border_width_percent = 0.2
    current_nbr_products = len(product_dico)
    mask = np.zeros((image.shape[0], image.shape[1]))
    end_y = image.shape[0]
    
    origin = os.path.basename(os.path.dirname(boxfile))
    sheet = pe.get_sheet(file_name=boxfile)
    
    lcol = 6
    if sheet[0,6] == 'UPC':
        lcol = 7
        
    nrows = sheet.number_of_rows()
    for row_idx in range(nrows-1):
        crow = sheet.row[row_idx+1]
        row = np.asarray(crow)
        
        if len(row) < 2:continue

        label = row[lcol]
        
        if section == None:
            print 'no section set when loading info'
            quit(0)
        
        
        if get_brand == 1:
            class_idx = gt.get_index_from_brand(section=section,brand=label)
            if class_idx == -1:
                print 'error loading info'
                quit(0)
        else:
            class_idx = 1
            
        x1 = float(row[2])
        if origin=='Origin-BottomLeft':
            y1 = end_y- float(row[3])
        else:
            y1 = float(row[3])
        x2 = x1 + float(row[4])
        y2 = y1 + float(row[5])
                    
        if x2 > image.shape[1]-1:
            x2 = image.shape[1]-1
        if y2 > image.shape[0]-1:
            y2 = image.shape[0]-1

        # write mask
        mask[y1:y2, x1:x2] = class_idx   
     #   mask[y1:y2, x1:x2] = 1        
        if load_borders == 1:
            db = np.round((x2-x1)*border_width_percent/2)
            if x2+db > image.shape[1]-1:
                x2 = image.shape[1]-1-db
            mask[y1:y2, x1-db:x1+db] = -1
            mask[y1:y2, x2-db:x2+db] = -1
        
        # save box info
        box = [y1,y2,x1,x2,label,class_idx]	
        #box = [y1,y2,x1,x2,label,1]	
        product_dico[current_nbr_products] = box
        current_nbr_products += 1
        
    return mask
    
# function to load brand info from xlsx file.
# It iteratively fills maxnames (current total number of brands) and lutnames (storing brand names)
# The mask is filled with the corresponding label for each box
def load_UPC_data_xls(image, boxfile, UPC_dict, product_dico, load_borders = 0):
    border_width_percent = 0.25
    current_nbr_products = len(product_dico)
    mask = np.zeros((image.shape[0], image.shape[1]))
    end_y = image.shape[0]
    
    origin = os.path.basename(os.path.dirname(boxfile))
    sheet = pe.get_sheet(file_name=boxfile)
    
    lcol = 6
    assert(sheet[0,6] == 'UPC')
    
    nrows = sheet.number_of_rows()
    for row_idx in range(nrows-1):
        crow = sheet.row[row_idx+1]
        row = np.asarray(crow)
        
        if len(row) < 2:continue

        label = row[lcol]
        #box = [row['C'],row['D'],row['E'],row['F']]
        if UPC_dict.has_key(label)==False:
            #print 'Warning, this label has never been seen before !', label
            UPC_dict[label] = len(UPC_dict)
            
        x1 = float(row[2])
        if origin=='Origin-BottomLeft':
            y1 = end_y- float(row[3])
        else:
            y1 = float(row[3])
        x2 = x1 + float(row[4])
        y2 = y1 + float(row[5])
                    
        if x2 > image.shape[1]-1:
            x2 = image.shape[1]-1
        if y2 > image.shape[0]-1:
            y2 = image.shape[0]-1

        # write mask
        mask[y1:y2, x1:x2] = UPC_dict[label]   
        if load_borders == 1:
            db = np.round((x2-x1)*border_width_percent/2)
            if x2+db > image.shape[1]-1:
                x2 = image.shape[1]-1-db
            mask[y1:y2, x1-db:x1+db] = -1
            mask[y1:y2, x2-db:x2+db] = -1
        
        # save box info
        box = [y1,y2,x1,x2,label,UPC_dict[label]]	
        product_dico[current_nbr_products] = box
        current_nbr_products += 1
        
    return mask    
    
def save_objects_in_xls(objects, save = '/output/visualisations/', convert_scores = 0):
    if convert_scores == 1:
        obj = dict()
        for o in objects:
            cbox = objects[o]
            if len(cbox) > 4: # include identification scores 
                outbox = [cbox[0], cbox[1], cbox[2], cbox[3], get_brand_per_label(cbox[4][0]), cbox[5][0], get_brand_per_label(cbox[4][1]), cbox[5][1], get_brand_per_label(cbox[4][2]), cbox[5][2], get_brand_per_label(cbox[4][3]), cbox[5][3], get_brand_per_label(cbox[4][4]), cbox[5][4]]
            else:
                outbox = [cbox[0], cbox[1], cbox[2], cbox[3]]
            
            obj[o] = outbox
        content = pe.utils.dict_to_array(obj)    
        sheet = pe.Sheet(content)
    else:
        content = pe.utils.dict_to_array(objects)    
        sheet = pe.Sheet(content)
    
    sheet.transpose()
    output = save + '_result.xls'
    sheet.save_as(output)
    print 'Saved xls result', output
