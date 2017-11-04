# imports
import numpy as np
import pandas as pd
import collections
import operator
from copy import copy
from fct_getters import get_brand_from_upc

# -

def verify_brands_2(image_brand, text_final, brand_correspondence, option_reco, option_plot, n_best):
    """
    Merges results from image and text and regarding brand recognition.

    Inputs:
    - image_brand = {origin_image_name: {idbox: [y1, y2, x1, x2, [brand_id], [brand_id_score]]}}
    - text_final = {filename: {idbox: [[[upc, confidence], ...], [[brand_id, confidence], ...]]}}
    - brand_correspondence = either [] or np.array() of size (x, 2) where x is the number of brand correspondences
    - option_reco = either "text_image" / "text_only" / "image_only"
    - option_plot = either "on" / "debug" / "off"
    - n_best = length of [brand_id] in output

    Outputs:
    - verify_brands_text_only = {origin_image_name: {idbox: [y1, y2, x1, x2, np.array([brand_id]), np.array([brand_id_score])]}}
    - verify_brands = {origin_image_name: {idbox: [y1, y2, x1, x2, np.array([brand_id]), np.array([brand_id_score])]}}
    """
    
    # In order to merge image and text results,
    # we make the following hypothesis:
    # - we can classify each brand in three groups, by degressive importance
    # --> 1) a brand that appears in image AND text
    # --> 2) a brand that appears in image
    # --> 3) a brand that appears in text
    # - in each of these groups we perform a relative classification based on either:
    # --> 'mean' of confidences for group (1)
    # --> 'max' of confidences for groups (2) and (3)
    # - whatever the outcome, brands from group (1) will always be 
    #   in front of those from group (2) and (3)
    # - structure of a group(x) = [(brand_id, brand_id_score)] sorted by decreasing confidence
    # - additionnaly, we create another group: group_text to classify all text results alone


    verify_brands_text_only = {}
    verify_brands = {}

    for origin_image_name, boxes in image_brand.items():
        
        for idbox, box_info in boxes.items():
            
            image_brand_dict = dict(zip(map(int, box_info[4]), box_info[5]))  # {brand_id: brand_id_score}  while converting brand_id to type int
            if origin_image_name in text_final and idbox in text_final[origin_image_name]:
                text_brand_dict = dict([(int(b[0]), b[1]) for b in text_final[origin_image_name][idbox][1]])  # {brand_id: brand_id_score}  while converting brand_id to type int
            else:
                text_brand_dict = {-1: -1 for i in range(n_best)}

            # some brands within image_brand_dict and text_brand_dict
            # may be related. E.g: 'L\'Oreal Paris' and 'L\'Oreal Kids'
            # We want to merge them to avoid repetition.
            if brand_correspondence != []:
                # image
                im_keys = list(image_brand_dict)[:]
                for brand_id in im_keys:
                    if brand_id in list(brand_correspondence[:, 0]):
                        correspondance_brand_id = int(brand_correspondence[list(brand_correspondence[:, 0]).index(brand_id), 1])
                        if correspondance_brand_id in image_brand_dict:
                            image_brand_dict[correspondance_brand_id] = max(image_brand_dict[brand_id], image_brand_dict[correspondance_brand_id])
                            image_brand_dict.pop(brand_id)  # replace brand_id key with correspondance_brand_id key
                        else:
                            image_brand_dict[correspondance_brand_id] = image_brand_dict.pop(brand_id)  # replace brand_id key with correspondance_brand_id key
                # text
                text_keys = list(text_brand_dict)[:]
                for brand_id in text_keys:
                    if brand_id in list(brand_correspondence[:, 0]):
                        correspondance_brand_id = int(brand_correspondence[list(brand_correspondence[:, 0]).index(brand_id), 1])
                        if correspondance_brand_id in text_brand_dict:
                            text_brand_dict[correspondance_brand_id] = max(text_brand_dict[brand_id], text_brand_dict[correspondance_brand_id])
                            text_brand_dict.pop(brand_id)  # replace brand_id key with correspondance_brand_id key
                        else:
                            text_brand_dict[correspondance_brand_id] = text_brand_dict.pop(brand_id)  # replace brand_id key with correspondance_brand_id key

            # group (1)
            group1 = []
            for brand_id in image_brand_dict:
                if brand_id in text_brand_dict and brand_id != -1:
                    # brand_id belongs to group1
                    group1.append((brand_id, (image_brand_dict[brand_id]+text_brand_dict[brand_id])/float(2)))
            group1.sort(key=lambda x: x[1], reverse=True)  # sort by decreasing confidence

            # group (2)
            group2 = []
            for brand_id in image_brand_dict:
                if brand_id not in (x[0] for x in group1) and brand_id != -1:
                    # brand_id belongs to group2
                    group2.append((brand_id, image_brand_dict[brand_id]))
            group2.sort(key=lambda x: x[1], reverse=True)

            # group (3)
            group3 = []
            for brand_id in text_brand_dict:
                if brand_id not in (x[0] for x in group1) and brand_id != -1:
                    # brand_id belongs to group3
                    group3.append((brand_id, text_brand_dict[brand_id]))
            group3.sort(key=lambda x: x[1], reverse=True)

            # additional group_text
            if origin_image_name in text_final and idbox in text_final[origin_image_name]:
                group_text = [(b, s) for b, s in text_brand_dict.items()]
                group_text.sort(key=lambda x: x[1], reverse=True)
            else:
                group_text = [(-1, -1) for i in range(n_best)]

            # get final brand vectors (fbv)
            fbv = group1 + group2 + group3

            # truncate to keep only n_best brands
            fbv = fbv[:n_best]

            # check if fbv is of length n_best. Else add as much (-1, -1) as necessary
            if len(fbv) < n_best:
                n = len(fbv)
                fbv += [(-1, -1) for i in range(n_best - n)]
            
            # update verify_brands
            if origin_image_name not in verify_brands:
                verify_brands[origin_image_name] = {}
            verify_brands[origin_image_name][idbox] = box_info[0:4] + [np.array([x[0] for x in fbv])] + [np.array([x[1] for x in fbv])]

            # proceed identically for verify_brands_text_only
            fbv_text = group_text[:n_best]
            if len(fbv_text) < n_best:
                n = len(fbv_text)
                fbv_text += [(-1, -1) for i in range(n_best - n)]
            if origin_image_name not in verify_brands_text_only:
                verify_brands_text_only[origin_image_name] = {}
            verify_brands_text_only[origin_image_name][idbox] = box_info[0:4] + [np.array([x[0] for x in fbv_text])] + [np.array([x[1] for x in fbv_text])]
    
    return [verify_brands_text_only, verify_brands]


def verify_UPC_2(image_upc, text_final, option_reco, option_plot, fam_id, skus_and_brands_for_fam_id, path_to_data, n_best):
    """
    Merges results from image and text and regarding UPC recognition.

    Inputs:
    - image_upc = {origin_image_name: {idbox: [y1, y2, x1, x2, [brand_id], [brand_id_score], [upc], [upc_score]]}}
    - text_final = {origin_image_name: {idbox: [[[upc, score], ...], [[brand_id, score], ...]]}}
    - option_reco = either "text_image" / "text_only" / "image_only"
    - option_plot = either "on" / "debug" / "off"
    - fam_id = 
    - skus_and_brands_for_fam_id = all brands and skus  for the fam_id  with the following structure : 
      {
       "brand_id_0" : 
                    {
                    "name" : "Pantene",
                    "child" : {
                               "sku_id_0" : "Name_sku_id_0",
                               "sku_id_1" : "Name_sku_id_1",
                               ...
                    }
     ...
     }
    - path_to_data =
    - n_best = length of [brand_id] in output

    Outputs:
    - verify_upc_text_only = {origin_image_name: {idbox: [y1, y2, x1, x2, np.array([upc]), np.array([upc_score])]}}
    - verify_upc = {origin_image_name: {idbox: [y1, y2, x1, x2, np.array([upc]), np.array([upc_score])]}}
    """


    verify_upc_text_only = {}
    verify_upc = {}


    for origin_image_name, boxes in image_upc.items():
        
        for idbox, box_info in boxes.items():
            
            image_upc_dict = dict(zip(box_info[6], box_info[7]))
            if origin_image_name in text_final and idbox in text_final[origin_image_name]:
                text_upc_dict = dict([(u[0], u[1]) for u in text_final[origin_image_name][idbox][0]])
            else:
                text_upc_dict = {-1: -1 for i in range(n_best)}

            # Whatever the upc stored in image_upc_dict and text_upc_dict,
            # we must keep in mind that a brand_id has already been chosen in each idbox.
            # All selected upcs must be related to this brand.
            # We filter image_upc_dict and text_upc_dict to comply.
            box_brand_id = box_info[4][0]  # brand_id with highest confidence

            # get all skus for this brand
            try:
                skus_available = [int(x) for x in skus_and_brands_for_fam_id[str(box_brand_id)]["child"].keys()]

            except Exception:
                skus_available = []
                pass

            # check image_upc_dict
            im_keys = list(image_upc_dict)[:]
            
            for upc in im_keys:
                if upc not in skus_available :
                    # upc doesn't comply -> delete upc
                    image_upc_dict.pop(upc, None)
            
            # check text_upc_dict
            text_keys = list(text_upc_dict)[:]
            for upc in text_keys:
                if upc not in skus_available:
                    # upc doesn't comply -> delete upc
                    text_upc_dict.pop(upc, None)

            # group (1)
            group1 = []
            for upc in image_upc_dict:
                if upc in text_upc_dict and upc != -1:
                    # upc belongs to group1
                    group1.append((upc, (image_upc_dict[upc]+text_upc_dict[upc])/float(2)))
            group1.sort(key=lambda x: x[1], reverse=True)  # sort by decreasing confidence
            
            # group (2)
            group2 = []
            for upc in image_upc_dict:
                if upc not in (x[0] for x in group1) and upc != -1:
                    # upc belongs to group2
                    group2.append((upc, image_upc_dict[upc]))
            group2.sort(key=lambda x: x[1], reverse=True)

            # group (3)
            group3 = []
            for upc in text_upc_dict:
                if upc not in (x[0] for x in group1) and upc != -1:
                    # upc belongs to group3
                    group3.append((upc, text_upc_dict[upc]))
            group3.sort(key=lambda x: x[1], reverse=True)

            # additional group_text
            if origin_image_name in text_final and idbox in text_final[origin_image_name]:
                group_text = [(u[0], u[1]) for u in text_final[origin_image_name][idbox][0]]
                group_text.sort(key=lambda x: x[1], reverse=True)
            else:
                group_text = [(-1, -1) for i in range(n_best)]

            # get final upc vectors (fuv)
            fuv = group1 + group2 + group3

            # truncate to keep only n_best brands
            fuv = fuv[:n_best]

            # check if fuv is of length n_best. Else add as much (-1, -1) as necessary
            if len(fuv) < n_best:
                n = len(fuv)
                fuv += [(-1, -1) for i in range(n_best - n)]

            # update verify_upc
            if origin_image_name not in verify_upc:
                verify_upc[origin_image_name] = {}
            verify_upc[origin_image_name][idbox] = box_info[0:4] + [np.array([x[0] for x in fuv])] + [np.array([x[1] for x in fuv])]

            # proceed identically for verify_upc_text_only
            fuv_text = group_text[:n_best]
            if len(fuv_text) < n_best:
                n = len(fuv_text)
                fuv_text += [(-1, -1) for i in range(n_best - n)]
            if origin_image_name not in verify_upc_text_only:
                verify_upc_text_only[origin_image_name] = {}
            verify_upc_text_only[origin_image_name][idbox] = box_info[0:4] + [np.array([x[0] for x in fuv_text])] + [np.array([x[1] for x in fuv_text])]
            
    return [verify_upc_text_only, verify_upc]


# unit tests

def test_brand():
    image_brand = {"im0.jpg": {0: [0, 10, 0, 10, ['6500', '6501', '6504'], [0.91, 0.46, 0.70]], 
                               1: [5, 15, 5, 15, ['6581', '6582', '6583'], [0.35, 0.46, 0.82]], 
                               2: [10, 20, 10, 20, ['6000', '6001', '6002', '6003', '6004'], [0.1, 0.2, 0.3, 0.4, 0.5]]}}
    text_final = {"im0.jpg": {0: [[['05000000000000', 0.5], ['05000000000004', 0.5]], [['6500', 0.5], ['6501', 0.6], ['6505', 0.3]]],
                              1: [[['05000000000001', 0.6], ['05000000000002', 0.6]], [['6581', 0.5], ['6583', 0.6], ['6584', 0.3]]],
                              2: [[['05000000000003', 0.6], ['05000000000004', 0.6]], [['6585', 0.5], ['6586', 0.6], ['6588', 0.3]]]}}
    brand_correspondence = np.array(pd.DataFrame({"brand_id_1": [6500, 6501], "brand_id_2": [6503, 6504]}))
    option_reco = "text_image"
    option_plot = "on"
    n_best = 10
    verify_brands_2(image_brand, text_final, brand_correspondence, option_reco, option_plot, n_best)


### FORMER VERSION ###
#
#


def verify_brands(image_brand, text_final, brand_correspondence, option_reco, option_plot,n_best):
    # image_brand = {filename: {idbox: [StartY, EndY, StartX, EndX, [brand_label], [brand_id_confidence], shelf_nb, ??]}}
    # text_final = {filename: {idbox: [[upc, upc_score], [brand, brand_score]]}}
    # brand_correspondence = DataFrame
    # output = {filename: {idbox: [StartY, EndY, StartX, EndX, [brand_id_list], [brand_id_confidence_list]]}}  ([brand_label] and [brand_label_confidence] are sorted against [upc_id_confidence])

    # 2 parameters: a (0<a<1), b (b>0)
    # confidence = a*X1^b + (1-a)*X2^b  (X1: TEXT, X2: IMAGE)
    # a: enable to give more importance to TEXT (high a) or IMAGE (low a) recognition
    # b: enable to give more importance to unbalanced couples (X1=1, X2=0 - high b) over balanced couples (X1=0.5, X2=0.5 - low b)

	
    # print "\n\n\nTEXT-FINAL"
    # print text_final
    # print "\n\n\nTEXT-FINAL"


    verify_brands = dict()
    verify_brands_text_only = dict()

    for filename, box_set in image_brand.items():
        
        verify_brands[filename] = dict()
        verify_brands_text_only[filename] =dict()
        
        for idbox, box_params in box_set.items():
        
            brand_final = collections.OrderedDict()  # key: brand_id | value: aggregated confidence
            brands_text_only = collections.OrderedDict()
			
            if len(text_final[filename])>0 : 
			
              for br in text_final[filename][idbox][1][::-1]:

                brand_id = int(br[0])
                brand_score = br[1]
				            
                if brand_correspondence != []:
                
                    try: # if brands are associated to other brands (specific requirement of clients); then link them
                        ind_corr = list(brand_correspondence[:, 0]).index(brand_id)
                        brand_id = int(brand_correspondence[ind_corr, 1])

                    except ValueError:
                       brand_id = brand_id


                if brand_score > 0.7:
                    print "brand score higher than 0.7"
                    brand_final[brand_id] = brand_score  # store results from TEXT
                    brands_text_only[brand_id] = brand_score  # store results from TEXT
					
            if option_reco=='text/image':
                
                for i, brand2 in enumerate(image_brand[filename][idbox][4]):
                
                    if brand_correspondence!=[]:
                
                        try: # if brands are associated to other brands (specific requirement of clients); then link them
                            ind_corr=list(brand_correspondence[:,0]).index(brand2)
                            brand2=brand_correspondence[ind_corr,1]
                        except ValueError:
                            brand2=brand2


                    ## Kevin - new condition on image confidence > 0.85 and 2/3 was replaced by 4/5
                    if image_brand[filename][idbox][5][i] > 0:
                        if brand2 in brand_final.keys():
                            print "Brand already detected by TEXT with condidence = %s" % brand_final[brand2]
                            if image_brand[filename][idbox][5][i] > brand_final[brand2] * 4/5: # if confidance of image is higher than confidance of text:
                                print "Brand confidence TEXT = %s replaced by confidence IMAGE = %s" % (brand_final[brand2], image_brand[filename][idbox][5][i])
                                brand_final[brand2] = image_brand[filename][idbox][5][i]
                                
                        else: # if brand has never seen before in the dictionary
                            brand_final[brand2] = image_brand[filename][idbox][5][i]
                    ## Kevin


                sorted_brand_final = sorted(brand_final.items(), key=operator.itemgetter(1), reverse=True)  # list of tupples
                brd = [sorted_brand_final[j][0] for j in range(0, len(sorted_brand_final))]
                brd_scores = [sorted_brand_final[k][1] for k in range(0, len(sorted_brand_final))]
                while len(brd) < n_best: 
                    brd.append(-1)
                    brd_scores.append(-1)
                # keep only first 6 results BRAND_ID / BRAND_ID_SCORE
                verify_brands[filename][idbox] = [box_params[0], box_params[1], box_params[2], box_params[3], np.array(brd[0:min(n_best, len(brd))]), np.array(brd_scores[0:min(n_best, len(brd_scores))])]

            # order brand_final by value (biggest confidence first)
            if option_reco == 'text_only' or (option_reco == 'text/image' and option_plot == 'on'):
     
                sorted_brand_final = sorted(brands_text_only.items(), key=operator.itemgetter(1), reverse=True)  # list of tupples

                brd = [sorted_brand_final[j][0] for j in range(0, len(sorted_brand_final))]
                brd_scores = [sorted_brand_final[k][1] for k in range(0, len(sorted_brand_final))]
                if brd==[]: brd=-1*np.ones(n_best)
                if brd_scores==[]: brd_scores=-1*np.ones(n_best)
                while len(brd) < n_best: 
                    brd.append(-1)
                    brd_scores.append(-1)
                verify_brands_text_only[filename][idbox] = [box_params[0], box_params[1], box_params[2], box_params[3], np.array(brd[0:min(n_best, len(brd))]), np.array(brd_scores[0:min(n_best, len(brd_scores))])]

    return [verify_brands_text_only, verify_brands]


def verify_UPC(image_upc, text_final, option_reco, option_plot,fam_id,fam_df,path_to_data,n_best):
    # image_upc = {filename: {idbox: [StartY, EndY, StartX, EndX, [brand_id_list], [brand_id_confidence_list], [upc_list], [upc_confidence_list]]}}
    # text_final = {filename: {idbox: [{upc: confidence}, {brand: confidence}]}}
    # verify_upc = {filename: {idbox: [StartY, EndY, StartX, EndX, [upc_list], [upc_confidence_list]]}}  ([upc_list] and [upc_confidence_list] are sorted from highest to lowest confidence)

    # 2 parameters: a (0<a<1), b (b>0)
    # confidence = a*X1^b + (1-a)*X2^b  (X1: TEXT, X2: IMAGE)
    # a: enable to give more importance to TEXT (high a) or IMAGE (low a) recognition
    # b: enable to give more importance to unbalanced couples (X1=1, X2=0 - high b) over balanced couples (X1=0.5, X2=0.5 - low b)

    verify_upc = dict()
    verify_upc_text_only = dict()

    #raw_input()
    for filename, box_set in image_upc.items():
        
        verify_upc[filename] = dict()
        verify_upc_text_only[filename] = dict()

        for idbox, box_params in box_set.items():
            
            upc_final = {}  # key : upc // value : aggregated confidence
            upc_text_only = {}
			
            if len(text_final[filename])>0 :           
			
              for u in text_final[filename][idbox][0]:
                upc = u[0]
                upc_score = u[1]
                brand_id = get_brand_from_upc(upc, fam_id, path_to_data+'/Picture_database/general_database_storage', df=fam_df)
                brand_id_selected = image_upc[filename][idbox][4][0]
                if brand_id == brand_id_selected and upc_score > 0.7:
                    upc_final[upc] = upc_score  # store results from TEXT
                    upc_text_only[upc] = upc_score  # store results from TEXT


            if option_reco == 'text/image':

                for i, upc2 in enumerate(image_upc[filename][idbox][6]):
                    if upc2 != -1:

                        if upc2 in upc_final:
                            if image_upc[filename][idbox][7][i] > upc_final[upc2] * 2/3:
                                
                                upc_final[upc2] = image_upc[filename][idbox][7][i]
                        else:
                            
                            upc_final[upc2] = image_upc[filename][idbox][7][i]

           
                # order upc_final by value (biggest confidence first)
                sorted_upc_final = sorted(upc_final.items(), key=operator.itemgetter(1), reverse=True)  # list of tupples
                u = [sorted_upc_final[j][0] for j in range(0, len(sorted_upc_final))]
                u_scores = [sorted_upc_final[k][1] for k in range(0, len(sorted_upc_final))]
                while len(u) < n_best: 
                    u.append(-1)
                    u_scores.append(-1)
                # only keep 5 first UPC / UPC_SCORES found
                verify_upc[filename][idbox] = [box_params[0], box_params[1], box_params[2], box_params[3], np.array(u[0:min(n_best, len(u))]), np.array(u_scores[0:min(n_best, len(u_scores))])]
                

            if option_reco == 'text_only' or (option_reco == 'text/image' and option_plot == 'on'):
                # order upc_final by value (biggest confidence first)
                sorted_upc_final = sorted(upc_text_only.items(), key=operator.itemgetter(1), reverse=True)  # list of tupples
                u = [sorted_upc_final[j][0] for j in range(0, len(sorted_upc_final))]
                u_scores = [sorted_upc_final[k][1] for k in range(0, len(sorted_upc_final))]
                if u == []: u = -1*np.ones(n_best)
                if u_scores == []: u_scores = -1*np.ones(n_best)
                while len(u) < n_best: 
                    u.append(-1)
                    u_scores.append(-1)
                verify_upc_text_only[filename][idbox] = [box_params[0], box_params[1], box_params[2], box_params[3], np.array(u[0:min(n_best, len(u))]), np.array(u_scores[0:min(n_best, len(u_scores))])]

    #raw_input()
    return [verify_upc_text_only, verify_upc]







