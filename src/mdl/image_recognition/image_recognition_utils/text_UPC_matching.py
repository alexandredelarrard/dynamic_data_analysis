# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 10:44:31 2016

@author: Kevin Serru
"""

import time
import re, collections
import numpy as np
import os
import pandas as pd
import sys
import time
from collections import OrderedDict
import COMMON

# - To test KEVIN

def text_UPC_matching(candidates, candidates_composed, keywords_brands, keywords_upcs, brand_id_keywords, fam_id, labels_1_2_for_label_0):
    """
    Associate a couple (brand, upc) to a group of words_detected in a product image (cf. text_to_keywords.py)
    
    Inputs:
    - candidates = {word: [[keyword, confidence], ...]}  # keywords sorted by decreasing confidence
    - candidates_composed = {word: [[keyword, confidence], ...]}  # keywords sorted by decreasing confidence
    - keywords_brands = DataFrame() word | brand_ids | brands (index: 'word')
    - keywords_upcs = DataFrame() word | upcs (index: 'word')
    - brand_id_keywords = {brand_id: [keywords]}  # e.g: {<Tech Universe id>: ['tech', 'universe']}
    - fam_id = id of family. E.g: '65010000'
    - dataframe = DataFrame() object. Workbook database of family fam_id
    
    Outputs:
    - [upc, brand_id]
    -> upc = [[upc, confidence], ...]  (decreasing confidence)
    -> brand_id = [[brand_id, confidence], ...]  (decreasing confidence)
    """
    
    brand_id = {}
    upc = {}
    
    # A word may be associated with a brand that's single worded (e.g: 'Logitech') or 
    # have more than one word (e.g: 'Tech Universe').
    # Thus we have two methods of word/keyword association: one that selects keywords linked to 
    # single worded brands, another that selects keywords linked to multiple-words brands.

    # While adding brand_ids/upcs to main dictionaries brand_id/upc by successive inference on keywords,
    # we might add an element that is already stored with a certain confidence.
    # To update this element, we can chose one of two methods: 'max' or 'sum'.
    # - 'max' method updates the element's confidence with the maximum between new confidence
    # and stored confidence.
    # - 'sum' method updates the element's confidence with the sum of the two confidences.
    # Default method is 'max'
    method = 'max'

    # We start with keywords related to brands that are single-worded
    brand_id = brand_single_selection(brand_id, candidates, keywords_brands, method)

    # We pursue with mutiple-word-brands keywords
    brand_id = brand_multiple_selection(brand_id, candidates_composed, brand_id_keywords, method)

    # In order to find UPCs, we do a cross selection between:
    # - upcs found with keywords
    # - upcs of brands found with keywords
    upc = upc_selection(upc, candidates, keywords_upcs, brand_id, method, labels_1_2_for_label_0, fam_id=fam_id)

    # When using 'sum' method, some element's confidence might grow higher than 1.
    # For compatibility reasons with the rest of the software, we have to normalize them.
    # Normalization rule: if confidence > 1: confidence = 1
    if method == 'sum':
        for b, c in brand_id.items():
            if c > 1:
                brand_id[b] = float(1)
        for u, c in upc.items():
            if c > 1:
                upc[u] = float(1)

    # Finally, we transform brand_id and upc into list of pairs [b/u, confidence], sorted by highest confidence
    final_brand_id = sorted([[int(b), c] for b, c in brand_id.items()], key=lambda x: x[1], reverse=True)
    final_upc = sorted([[u, c] for u, c in upc.items()], key=lambda x: x[1], reverse=True)

    return [final_upc, final_brand_id]


# --- utils text_upc_matching --- #


def brand_single_selection(brand_id_selection, candidates, keywords_brands, method):
    """
    Selects single-word brands by keyword inference.

    Inputs:
    - brand_id_selection = [[brand_id, highest_confidence], ...]  (decreasing confidence)
    - candidates = {word: [[keyword, confidence], ...]}  # keywords sorted by decreasing confidence
    - keywords_brands = DataFrame() word | brand_ids | brands (index: 'word')
    - method = 'max' (default) / 'sum'

    Output:
    - brand_id_selection = [[brand_id, highest_confidence], ...]  (decreasing confidence)  # updated
    """

    for word, keywords in candidates.items():
        for keyword_info in keywords:
            keyword = keyword_info[0]
            confidence = keyword_info[1]
            
            # Get all brand ids associated with keyword and update brand_id_selection
            if keyword in keywords_brands.index:
                brand_ids = extract_list_from_string(keywords_brands.loc[keyword, 'brand_ids'], dtype=int)
                brand_id_selection = selection_function(brand_id_selection, brand_ids, confidence, method=method)
    
    return brand_id_selection


def brand_multiple_selection(brand_id_selection, candidates_composed, brand_id_keywords, method):
    """
    Selects multiple-word brands by keyword inference.

    Inputs:
    - brand_id_selection = [[brand_id, highest_confidence], ...]  (decreasing confidence)
    - candidates_composed = {word: [[keyword, confidence], ...]}  # keywords sorted by decreasing confidence
    - brand_id_keywords = {brand_id: [keywords]}  # e.g: {<Tech Universe id>: ['tech', 'universe']}
    - method = 'max' (default) / 'sum'

    Output:
    - brand_id_selection = [[brand_id, highest_confidence], ...]  (decreasing confidence) # updated
    """

    # From candidates_composed we get lists of keyword/confidence couples. Putting all these
    # lists together we build a super_dict of keyword/confidence couples. If some keywords are listed more than
    # once, with different confidence, we associate to the keyword the maximum of all confidences.
    # E.g: candidates_composed = {'word': [['worm', 72], ['world', 60], ..., ['potato', 5]], 
    #                             'worn': [['worm', 83], ['world', 54], ..., ['potato', 4]],
    #                             'potent: [['potato', 56], ...]'}
    # super_dict = {'potato': 56, 'worm': 83, 'world': 60, ...}

    super_dict = {}
    for word, keywords_info in candidates_composed.items():
        for keyword_info in keywords_info:
            keyword = keyword_info[0]
            confidence = keyword_info[1]
            if keyword in super_dict:
                max_confidence = max(super_dict[keyword], confidence)
                super_dict[keyword] = max_confidence
            else:
                super_dict[keyword] = confidence

    # We want to check if keywords in super_dict (all keywords associated with words found in the image)
    # match any of the keywords related to composed brand ids. If all keywords related to a brand_id are 
    # found in super_dict, we determine a global confidence associated with brand_id by averaging the individual confidence
    # of all keywords.
    for brand_id, keywords in brand_id_keywords.items():
        if set(keywords).issubset(set(super_dict)):
            brand_id_confidence = sum((super_dict[keyword] for keyword in keywords)) / float(100 * len(keywords))

            # We update the brand_id_selection using either method 'max' (default) / 'sum'
            if brand_id in brand_id_selection:
                if method == 'max':
                    max_confidence = max(brand_id_selection[brand_id], brand_id_confidence)
                    brand_id_selection[brand_id] = max_confidence
                elif method == 'sum':
                    brand_id_selection[brand_id] += brand_id_confidence
            else:
                brand_id_selection[brand_id] = brand_id_confidence

    return brand_id_selection


def upc_selection(upc_selection, candidates, keywords_upcs, brand_id_selection, method, labels_1_2_for_label_0, fam_id=None):
    """
    Select UPCs by cross-validating results from brand_single_selection/brand_multiple_selection and keywords inference

    Inputs:
    - upc_selection = [[upc, highest_confidence], ...]  (decreasing confidence)
    - candidates = {word: [[keyword, confidence], ...]}  # keywords sorted by decreasing confidence
    - keywords_upcs = DataFrame() word | upcs (index: 'word')
    - brand_id_selection = [[brand_id, highest_confidence], ...]  (decreasing confidence)
    - method = either 'max' (default) or 'sum'
    - fam_id = id of family. E.g: '65010000'
    - labels_1_2_for_label_0 : all skus and brands for the fam_id

    Output:
    - upc_selection = [[upc, highest_confidence], ...]  (decreasing confidence)  # updated
    """

    # Find all possible upcs from brand_id previously selected
    upcs_from_brand_ids = []

    for brand_id in brand_id_selection:
        if str(brand_id) in labels_1_2_for_label_0 :
           if "child" in  labels_1_2_for_label_0[str(brand_id)] : 
               upcs_from_brand_ids += labels_1_2_for_label_0[str(brand_id)]["child"].keys()

    # Find all possible upcs from keywords
    for word, keywords in candidates.items():
        for keyword_info in keywords:
            keyword = keyword_info[0]
            confidence = keyword_info[1]
            # Get all upcs associated with keyword
            if keyword in keywords_upcs.index:
                upcs_from_words = extract_list_from_string(keywords_upcs.loc[keyword, 'upcs'])
                # Intersect two upc groups
                upcs_inf = list(set(upcs_from_brand_ids) & set(upcs_from_words))
                # Update upc_selection
                upc_selection = selection_function(upc_selection, upcs_inf, confidence, method=method)

    return upc_selection


def selection_function(selection, candidates, candidates_confidence, apply_division=True, method='max'):
    """
    Adds candidates to selection while stamping them with a "confidence" tag.
    Typically used to build a ranking system for brands and UPCs infered from words found in 
    product boxes via OCR.
    Inputs:
    - selection = DICT {element: confidence}  # contains former selected candidates
    - candidates = LIST [element]  # new candidates to be included in selection
    - candidates_confidence = INT confidence  # confidence associated with the batch of new candidates (in interval [0, 100])
    - apply_division = BOOL  # should we take into account the number of candidates and compute an "average confidence"
    - method = either 'max' (default) or 'sum'. See description below.

    Output:
    - selection = DICT {element: confidence}  # input updated
    """


    # The more candidates there is, the less confidence there can be in each one of them
    n = len(candidates) if apply_division else 1

    # We are going to include every candidate in the selection variable, along with a certain confidence
    # that we will need to compute according to some method.
    for element in candidates:
        
        # compute the new confidence, taking into account the number of candidates
        element_confidence = candidates_confidence / float(n * 100)

        if method == 'max':
            
            # If the candidate element was not in the selection set, or
            # if it was AND its now associated confidence is greater than the one previously stored
            # then we reset this element's confidence to the new value.
            # Thus we store only the MAX confidence associated with a selected element.
            if (element not in selection) or (element in selection and element_confidence > selection[element]):
                selection[element] = element_confidence

        elif method == 'sum':

            # If the candidate element was already in the selection set, we
            # SUM its confidence to the previously stored confidence.
            if element not in selection:
                selection[element] = 0
            selection[element] += element_confidence

    return selection


def extract_list_from_string(s, dtype=int):
    """
    Example:
    input = "['element0', 'element1', 'element2']"
    output = [element0, element1, element2]  where type(elementX) == dtype
    """
    r = [w.strip("'").strip('"') for w in s.strip('[]').split(', ')]
    if dtype == int: return [int(float(i)) for i in r]
    if dtype == str: return [str(i) for i in r]     


def keywords_generator(item):
    blacklist = ['count', 'ounce', 'oz', 'jr', 'each', 'free']
    item_k = item.replace('/', ' ').replace('-', ' ')
    item_k = item_k.lower()
    item_k = re.sub(r'[?|$|.|!]', r'', item_k)
    item_k = re.sub(r'[^a-zA-Z0-9 ]', r'', item_k)
    item_k = item_k.split()
    final_set = []
    for k in item_k:
        if not (k.isdigit() or k in blacklist):
            final_set.append(k)
    
    return final_set

