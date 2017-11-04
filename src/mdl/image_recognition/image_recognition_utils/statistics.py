# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:33:03 2016

@author: ben
"""
import COMMON as shared_pathes
import sys
sys.path.append(shared_pathes.path_to_utils)
import fct_getters as gt

# Function that returns the full shelf position given a list of products
# Shelf is measured with min/max position of found products
def get_full_shelf_coords(products):
    xmin = 100000
    ymin = 100000
    xmax = 0
    ymax = 0

    for p in products:
        pinfo = products[p]
        if pinfo[0] < ymin:
            ymin = pinfo[0]
        if pinfo[1] > ymax:
            ymax = pinfo[1]
        if pinfo[2] < xmin:
            xmin = pinfo[2]
        if pinfo[3] > xmax:
            xmax = pinfo[3]
    
    return [ymin, ymax, xmin, xmax]


# Function that returns the position of each shelf given a list of products
# Shelf is measured with min/max position of found products
def get_separate_shelf_coords(products):
    list_out = dict()
    
    for p in products:
        pinfo = products[p]
        shelf_idx = pinfo[6] 
        if list_out.has_key(shelf_idx) :
            if pinfo[0] < list_out[shelf_idx][0]:
                list_out[shelf_idx][0] = pinfo[0]
            if pinfo[1] > list_out[shelf_idx][1]:
                list_out[shelf_idx][1] = pinfo[1]
            if pinfo[2] < list_out[shelf_idx][2]:
                list_out[shelf_idx][2] = pinfo[2]
            if pinfo[3] > list_out[shelf_idx][3]:
                list_out[shelf_idx][3] = pinfo[3]
        else:
            list_out[shelf_idx] = [pinfo[0], pinfo[1], pinfo[2], pinfo[3]] 
    
    return list_out


# Function that returns the face counts per brand given a list of products
def get_nb_products_per_brand(products,section,brands_sheet):
    list_out = dict()
    
    for p in products:
        pinfo = products[p]
        brand = gt.get_brand_from_index(pinfo[4][0],section, brands_sheet)

        if list_out.has_key(brand):
            list_out[brand] += 1
        else :
            list_out[brand] = 1
    return list_out

# Function that returns the area repartition per brand
def get_brand_share(products,section,brands_sheet):
    full_area_products = 0
    list_out = dict()
    
    for p in products:
        pinfo = products[p]
        full_area_products += (pinfo[3] - pinfo[2])*(pinfo[1] - pinfo[0])
        brand = gt.get_brand_from_index(pinfo[4][0],section,brands_sheet)
        if list_out.has_key(brand): # 4 ou 5 ici ?!!!!!!
            list_out[brand] += (pinfo[3] - pinfo[2])*(pinfo[1] - pinfo[0])
        else :
            list_out[brand] = (pinfo[3] - pinfo[2])*(pinfo[1] - pinfo[0])
        
    for l in list_out:
        list_out[l] = list_out[l]*1.0/full_area_products
    
    return list_out

# Function that returns the empty space in all shelves
def get_empty_space(products):
    shelf_coords = get_full_shelf_coords(products)
    full_area_products = 0
    shelf_area = 1.0*(shelf_coords[3] - shelf_coords[2])*(shelf_coords[1] - shelf_coords[0])
    for p in products:
        pinfo = products[p]
        full_area_products += (pinfo[3] - pinfo[2])*(pinfo[1] - pinfo[0])
    
    area_out = (shelf_area - full_area_products)/shelf_area 
    return area_out

