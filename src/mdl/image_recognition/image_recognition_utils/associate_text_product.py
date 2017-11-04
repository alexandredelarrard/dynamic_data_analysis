# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 21:40:57 2016

@author: Kevin Serru
"""

import os
import xlsxwriter
from math import sqrt, acos, pi





def associate_text_product(text_in_file, ref_segmentation, path_save_text_found):
    """
    Associate text found in images (TEXT RECOGNITION) with product boxes found in images (SEGMENTATION)

    Inputs:
    - text_in_file = {origin_image_name: [[word, x1, y1, x2, y2]]}
    - ref_segmentation = {origin_image_name: {idBox: [y1, y2, x1, x2, Id_vertical]} -> idBox : from 0 to N_boxes-1
    - path_save_text_found = .../process/text_in_images/<name>.xlsx

    Output:
    - [dict_boxes, dict_overlapping_words]
    -> dict_boxes = {origin_image_name: {idBox: [x, y, w, h, [words], [[x1, y1, x2, y2]], [-1, word_index, ...]]}  # [-1, word_index,...] -1 if word is entirely contained in idBox, else (overlapping) idWord
    -> dict_overlapping_words = {origin_image_name: {word_index: [word, x1, y1, x2, y2, [idBox, area]]}
    """

    dict_boxes = {}
    dict_overlapping_words = {}

    for origin_image_name, words in text_in_file.items():

        dict_boxes[origin_image_name] = {}
        dict_overlapping_words[origin_image_name] = {}

        for word_index, word_info in enumerate(words):

            word = word_info[0]
            w_x1 = word_info[1]
            w_y1 = word_info[2]
            w_x2 = word_info[3]
            w_y2 = word_info[4]
            w_w = w_x2 - w_x1
            w_h = w_y2 - w_y1

            # We want to know to which product box the word box overlaps the most.
            # we create a vector good_box = {'idBox': idBox, 'area': percent_area}
            # that will be updated during a loop on every box
            overlapping_box = {'idBox':-1, 'area':-1}

            for idBox, product_info in ref_segmentation[origin_image_name].items():  # for each box in the image

                p_x1 = product_info[2]
                p_y1 = product_info[0]
                p_w = product_info[3] - product_info[2]
                p_h = product_info[1] - product_info[0]

                # - check overlapp - #
                left = max(w_x1, p_x1)
                right = min(w_x2, p_x1 + p_w)
                top = max(w_y1, p_y1)
                bottom = min(w_y2, p_y1 + p_h)
                area = (right - left) * (bottom - top)

                overlapping_condition = (area > 0 and (left < right) and (top < bottom))
                maximum_condition = (area > overlapping_box['area'])

                if overlapping_condition:
                    # update dict_overlapping_words
                    if word_index not in dict_overlapping_words[origin_image_name]:
                        dict_overlapping_words[origin_image_name][word_index] = [word, w_x1, w_y1, w_x2, w_y2]
                    dict_overlapping_words[origin_image_name][word_index].append([idBox, area])

                if overlapping_condition and maximum_condition:
                    # update overlapping_box
                    overlapping_box['idBox'] = idBox
                    overlapping_box['area'] = area

                if idBox not in dict_boxes[origin_image_name]:
                    dict_boxes[origin_image_name][idBox] = [p_x1, p_y1, p_w, p_h, [], [], []]

            # update dict_boxes
            if overlapping_box['idBox'] != -1:
                dict_boxes[origin_image_name][overlapping_box['idBox']][4].append(word)
                dict_boxes[origin_image_name][overlapping_box['idBox']][5].append([w_x1, w_y1, w_x2, w_y2])
                dict_boxes[origin_image_name][overlapping_box['idBox']][6].append(-1 if overlapping_box['area'] == 1 else word_index)

            # update dict_overlapping_words
            # -> sort [idBox, area] by decreasing percent_area
            if word_index in dict_overlapping_words[origin_image_name]:
                a = dict_overlapping_words[origin_image_name][word_index][5:]
                a.sort(key=lambda x: x[1], reverse=True)
                dict_overlapping_words[origin_image_name][word_index][5:] = a[0:2]  # only keep max 2 overlapping boxes

    # save text_in_file in a .xlsx file in /process/text_in_images
    #save_text_in_file(text_in_file,path_save_text_found)

    # if dictionary is empty, then fill it with empty dict for each picture
    if bool(dict_boxes) == False:
       for origin_image_name in ref_segmentation:
          dict_boxes[origin_image_name] = {}

    return [dict_boxes, dict_overlapping_words]


def associate_text_product_bonus(text_in_file, ref_segmentation,path_save_text_found):
    """
    Similar to associate_text_product plus takes into account overlapping words
    :param text_in_file: {} key: filename // value: [[word, X1, Y1, X2, Y2], [...], ...]
    :param ref_segmentation: {} key: filename // value: {idBox : [startY, endY, startX, endX, Id_vertical, ...]} -> idBox : from 0 to N_boxes-1
    :return: new_dict_boxes: {} key: filename // value: { idBox : [x, y, w, h, [words], [[X1, Y1, X2, Y2],...], [-1, idWord,...]]} -> [-1, idWord,...] (if word is not overlapping, -1, else idWord)
    """
    [dict_boxes, dict_overlapping_words] = associate_text_product(text_in_file, ref_segmentation,path_save_text_found)
    [one_box, two_boxes] = map_box_word_ovlp(dict_boxes, dict_overlapping_words)
    new_dict_boxes = one_box_to_new_dict_boxes(dict_boxes, dict_overlapping_words, one_box)
    new_dict_boxes = two_boxes_to_new_dict_boxes(new_dict_boxes, dict_overlapping_words, two_boxes)
    new_ref_segmentation = new_ref_from_new_dict(new_dict_boxes, ref_segmentation)

    return [new_dict_boxes, new_ref_segmentation]

def new_ref_from_new_dict(new_dict_boxes, ref_segmentation):
    """
    Build a new ref_segmentation from former ref_segmentation and new product box frontiers
    :param new_dict_boxes: {} key: filename // value: { idBox : [x, y, w, h, [words], [[X1, Y1, X2, Y2],...], [-1, idWord,...]]} -> [-1, idWord,...] (if word is not overlapping, -1, else idWord)
    :param ref_segmentation: {} key: filename // value: {idBox : [startY, endY, startX, endX, Id_vertical, ...]} -> idBox : from 0 to N_boxes-1
    :return: new_ref_segmentation : {} key: filename // value: {idBox : [startY, endY, startX, endX, Id_vertical, ...]} -> idBox : id of boxes remaining
    """
    new_ref_segmentation = {}
    for filename in new_dict_boxes.keys():
        new_ref_segmentation[filename] = {}
        for idBox in new_dict_boxes[filename].keys():
            bd = new_dict_boxes[filename][idBox][0:4]  # new box dimensions
            new_ref_segmentation[filename][idBox] = [bd[1], bd[1]+bd[3], bd[0], bd[0]+bd[2],ref_segmentation[filename][idBox][4],ref_segmentation[filename][idBox][5],ref_segmentation[filename][idBox][6],ref_segmentation[filename][idBox][7]]  # update dimension [startY, endY, startX, endX]
            #new_ref_segmentation[filename][idBox].append(ref_segmentation[filename][idBox][4:])  # update other info
    return new_ref_segmentation


def save_text_in_file(text_in_file, save_path):
    """
    Save text_in_file in the form of a .xlsx file
    :param text_in_file: {} key: filename // value: [[word, X1, Y1, X2, Y2], [...], ...]
    :param save_path: PATH to which to save .xlsx file
    :return: None
    """
    for filename in text_in_file.keys():
        # /!\ if there already was a workbook for a specific filename, it will be rewritten. /!\
        #print(save_path+'/'+filename+'.xlsx')
        workbook = xlsxwriter.Workbook(save_path+'/'+filename+'.xlsx')  # create workbook
        sh = workbook.add_worksheet("text")  # create worksheet
        # create all column headers
        sh.write(0, 0, "filename")
        sh.write(0, 1, "word")
        sh.write(0, 2, "x1")
        sh.write(0, 3, "y1")
        sh.write(0, 4, "x2")
        sh.write(0, 5, "y2")
        # fill all columns
        for row in range(1, len(text_in_file[filename])+1):
            col = 0
            sh.write(row, col, str(filename))
            col += 1
            sh.write(row, col, text_in_file[filename][row-1][0])
            col += 1
            sh.write(row, col, text_in_file[filename][row-1][1])
            col += 1
            sh.write(row, col, text_in_file[filename][row-1][2])
            col += 1
            sh.write(row, col, text_in_file[filename][row-1][3])
            col += 1
            sh.write(row, col, text_in_file[filename][row-1][4])
        workbook.close()


def map_box_word_ovlp(dict_boxes, dict_overlapping_words):
    """
    Build two dictionaries:
    - one identifies boxes with words that overlaps exclusively on them
    - second identifies couples of boxes with words that overlap on both of them
    :param dict_boxes: {} key: filename // value: { idBox : [x, y, w, h, [words], [[X1, Y1, X2, Y2],...], [-1, idWord,...]]} -> [-1, idWord,...] (if word is not overlapping, -1, else idWord)
    :param dict_overlapping_words: {} key: filename // value: { idWord : [word, X1, Y1, X2, Y2, [idBox, percentage_of_overlapping],...] }
    :return: [one_box, two_boxes]
     one_box: {} key: filename // value: {idBox: [idword1, idword2,...]}
     two_boxes: {} key: filename // value: {(idBox1, idBox2): [idword1, idword2,...]}
    """
    one_box = dict()  # key: filename // value: {idBox: [idword1, idword2,...]}
    two_boxes = dict()  # key: filename // value: {(idBox1, idBox2): [idword1, idword2,...]}
    word_couples = dict()
    for filename in dict_boxes.keys():
        two_boxes[filename] = dict()
        one_box[filename] = dict()
        word_couples[filename] = dict()
        for idword in dict_overlapping_words[filename].keys():

            if len(dict_overlapping_words[filename][idword])-5 == 1:  # the word overlaps one box only
                box_percent = dict_overlapping_words[filename][idword][5]  # box_percent = [idBox, percent_area]
                if box_percent[0] not in one_box[filename].keys():
                    one_box[filename][box_percent[0]] = [idword]
                else:
                    one_box[filename][box_percent[0]].append(idword)
            else:  # the word overlaps two boxes
                two_boxes_percent = dict_overlapping_words[filename][idword][5:7]
                b2 = (two_boxes_percent[0][0], two_boxes_percent[1][0])
                b2 = tuple(sorted(b2))  # (idBox1, idBox2)
                if b2 not in two_boxes[filename].keys():
                    two_boxes[filename][b2] = [idword]
                else:
                    two_boxes[filename][b2].append(idword)

    return [one_box, two_boxes]


def one_box_to_new_dict_boxes(dict_boxes, dict_overlapping_words, one_box):
    """
    Modify dict_boxes -> when a word overlaps only one box, adapt box dimensions to include the whole word
    :param dict_boxes: {} key: filename // value: { idBox : [x, y, w, h, [words], [[X1, Y1, X2, Y2],...], [-1, idWord,...]]} -> [-1, idWord,...] (if word is not overlapping, -1, else idWord)
    :param dict_overlapping_words: {} key: filename // value: { idWord : [word, X1, Y1, X2, Y2, [idBox, percentage_of_overlapping],...] }
    :param one_box: {} key: filename // value: {idBox: [idword1, idword2,...]}
    :return: new_dict_boxes
     new_dict_boxes: {} key: filename // value: { idBox : [x, y, w, h, [words], [[X1, Y1, X2, Y2],...], [-1, idWord,...]]} -> [-1, idWord,...] (if word is not overlapping, -1, else idWord)
     new_dict_overlapping_words: {} key: filename // value: { idWord : [word, X1, Y1, X2, Y2, [idBox, percentage_of_overlapping],...] }
    """
    new_dict_boxes = dict_boxes

    for filename in dict_boxes.keys():
        for idBox, idwords in one_box[filename].items():
            for idw in idwords:
                if dict_overlapping_words[filename][idw][5][1] != 1:  # if word is not entirely in a box
                    # update dimensions
                    wd = dict_overlapping_words[filename][idw][1:5]  # [x1, y1, x2, y2]
                    din = xywh_to_xyxy(new_dict_boxes[filename][idBox][0:4])  # [x1, y1, x2, y2]

                    # case 1 : box on the left
                    if din[0]<=wd[0] and din[2]>=wd[0]:
                        din[2] = wd[2]

                    # case 2 : box on the right
                    elif din[0]<=wd[2] and din[2]>=wd[2]:
                        din[0] = wd[0]

                    new_dict_boxes[filename][idBox][0:4] = xyxy_to_xywh(din)

                    # update words in box
                    # No need because they are already referenced in the box.


    return new_dict_boxes


def two_boxes_to_new_dict_boxes(dict_boxes, dict_overlapping_words, two_boxes):
    """
    Modify dict_boxes -> when a word overlaps two boxes, apply the following rule: add all the words overlapping lengths
    along the x-axis inside both boxs. The box that gets the more lenght gets all the words ("winner box"). We adapt the
    dimensions of the two boxes so that the winner box include all the words and the "loser box" exclude all the words.
    :param dict_boxes: {} key: filename // value: { idBox : [x, y, w, h, [words], [[X1, Y1, X2, Y2],...], [-1, idWord,...]]} -> [-1, idWord,...] (if word is not overlapping, -1, else idWord)
    :param dict_overlapping_words: {} key: filename // value: { idWord : [word, X1, Y1, X2, Y2, [idBox, percentage_of_overlapping],...] }
    :param two_boxes: {} key: filename // value: {(idBox1, idBox2): [idword1, idword2,...]}
    :return: [new_dict_boxes, new_dict_overlapping_words]
     new_dict_boxes: {} key: filename // value: { idBox : [x, y, w, h, [words], [[X1, Y1, X2, Y2],...], [-1, idWord,...]]} -> [-1, idWord,...] (if word is not overlapping, -1, else idWord)
     new_dict_overlapping_words: {} key: filename // value: { idWord : [word, X1, Y1, X2, Y2, [idBox, percentage_of_overlapping],...] }
    """
    new_dict_boxes = dict_boxes

    for filename in dict_boxes.keys():

        for duo_boxes, idwords in two_boxes[filename].items():

            # compute total word length in box 1 and 2
            db1 = new_dict_boxes[filename][duo_boxes[0]][0:4]  # [x, y, w, h]
            db2 = new_dict_boxes[filename][duo_boxes[1]][0:4]  # [x, y, w, h]
            lb1 = 0  # init word length in box 1
            lb2 = 0  # init word length in box 2
            max_wd = dict_overlapping_words[filename][idwords[0]][1:5]  # init "max dimension" of all the words ovlp b1 and b2 i.e. [min(x1), min(y1), max(x2), max(y2)]
            for idword in idwords:
                wd = dict_overlapping_words[filename][idword][1:5]  # [x1, y1, x2, y2]
                max_wd = [min(max_wd[0], wd[0]), min(max_wd[1], wd[1]), max(max_wd[2], wd[2]), max(max_wd[3], wd[3])]
                lb1 += max(0, min(wd[2], db1[0]+db1[2]) - max(wd[0], db1[0]))
                lb2 += max(0, min(wd[2], db2[0]+db2[2]) - max(wd[0], db2[0]))


            # update boxes dimensions and words
            if lb1 >= lb2:  # box 1 wins all the words
                # adapt dimensions of lb1 and lb2 so that lb1 [and lb2 doesn't] contains all ovlp words
                [d1, d2, box_disappear] = adjust_boxes_dimensions(dbin=db1, dbex=db2, max_wd=max_wd)

                # add/remove corresponding words
                new_dict_boxes = adapt_words(new_dict_boxes, dict_overlapping_words, filename, duo_boxes[0], duo_boxes[1], idwords)

                new_dict_boxes[filename][duo_boxes[0]][0:4] = d1  # assign new dimensions to box 1
                new_dict_boxes[filename][duo_boxes[1]][0:4] = d2  # assign new dimensions to box 2

                if box_disappear:  # make box 2 disappear from new_dict_boxes
                    new_dict_boxes[filename].pop(duo_boxes[1], None)

            else:  # box 2 wins all the words
                # adapt dimensions of lb1 and lb2 so that lb2 [and lb1 doesn't] contains all ovlp words
                [d2, d1, box_disappear] = adjust_boxes_dimensions(dbin=db2, dbex=db1, max_wd=max_wd)

                # add/remove corresponding words
                new_dict_boxes = adapt_words(new_dict_boxes, dict_overlapping_words, filename, duo_boxes[1], duo_boxes[0], idwords)

                new_dict_boxes[filename][duo_boxes[0]][0:4] = d1  # assign new dimensions to box 1
                new_dict_boxes[filename][duo_boxes[1]][0:4] = d2  # assign new dimensions to box 2

                if box_disappear:  # make box 1 disappear from new_dict_boxes
                    new_dict_boxes[filename].pop(duo_boxes[0], None)

    return new_dict_boxes


def adjust_boxes_dimensions(dbin, dbex, max_wd):
    """
    Adjusts coordinates of box 1 and 2 so that box 1 [and box 2 doesn't]
    contains all ovlp words whose dimensions are expressed in max_wd.
    :param dbin: [x, y, w, h] dimensions of box that must contain max_wd
    :param dbex: [x, y, w, h] dimensions of box that must exclude max_wd
    :param max_wd: [x1, y1, x2, y2] max dimensions of ovlp words
    :return: [din, dex]
    din: [x, y, w, h] new coordinates of box db1
    dex: [x, y, w, h] new coordinates of box db2
    """

    din = xywh_to_xyxy(dbin)  # box which includes
    dex = xywh_to_xyxy(dbex)  # box which excludes
    dex_disappear = 0

    # case 1 : box in on the left // ex on the right
    if din[0]<=max_wd[0] and din[2]>=max_wd[0] and dex[0]<=max_wd[2] and dex[2]>=max_wd[2]:
        din[2] = max_wd[2]
        dex[0] = max_wd[2]

    # case 2 : box ex on the left // in on the right
    elif dex[0]<=max_wd[0] and dex[2]>=max_wd[0] and din[0]<=max_wd[2] and din[2]>=max_wd[2]:
        dex[2] = max_wd[0]
        din[0] = max_wd[0]

    if dex[0] >= dex[2]:
        dex_disappear = 1

    return [xyxy_to_xywh(din), xyxy_to_xywh(dex), dex_disappear]


def xyxy_to_xywh(dim):
    """
    Transform [x1, y1, x2, y2] into [x, y, w, h]
    :param dim: [x1, y1, x2, y2]
    :return: [x, y, w, h]
    """
    return [dim[0], dim[1], dim[2]-dim[0], dim[3]-dim[1]]

def xywh_to_xyxy(dim):
    """
    Transform [x, y, w, h] into [x1, y1, x2, y2]
    :param dim: [x, y, w, h]
    :return: [x1, y1, x2, y2]
    """
    return [dim[0], dim[1], dim[0]+dim[2], dim[1]+dim[3]]

def adapt_words(dict_boxes, dict_overlapping_words, filename, idBox1, idBox2, idwords):
    """
    Add to box1 and remove from box2 corresponding words
    :param dict_boxes:
    :param dict_overlapping_words:
    :param filename:
    :param idBox1:
    :param idBox2:
    :param idwords:
    :return:
    """

    new_dict_boxes = dict_boxes

    for idword in idwords:
        if idword not in new_dict_boxes[filename][idBox1][6]:  # if idword not in [-1, idword, ...]
            # add all the missing words to box1
            new_dict_boxes[filename][idBox1][4].append(dict_overlapping_words[filename][idword][0])
            new_dict_boxes[filename][idBox1][5].append(dict_overlapping_words[filename][idword][1:5])
            new_dict_boxes[filename][idBox1][6].append(idword)

        if idword in new_dict_boxes[filename][idBox2][6]:  # if idword in [-1, idword, ...]
            # remove all words from box2
            index = new_dict_boxes[filename][idBox2][6].index(idword)
            del new_dict_boxes[filename][idBox2][4][index]
            del new_dict_boxes[filename][idBox2][5][index]
            del new_dict_boxes[filename][idBox2][6][index]

    return new_dict_boxes


class Point():

    x = None
    y = None

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return "%6.1f, %6.1f" % (self.x, self.y)

    def __eq__(self, obj):
        return obj.x == self.x and obj.y == self.y

    def distance_to_point(self, p):
        return sqrt((self.x-p.x)**2+(self.y-p.y)**2)

    def faces_line(self, line):
        return point_faces_edge(line, self)


class Rect():

    # Screen coordinates
    l_top  = None
    r_top  = None
    l_bot  = None
    r_bot  = None
    center = None
    width  = None
    height = None

    def __init__(self, x, y, width, height):
        assert width>0
        assert height>0
        self.l_top  = Point(x, y)
        self.r_top  = Point(x+width, y)
        self.r_bot  = Point(x+width, y+height)
        self.l_bot  = Point(x, y+height)
        self.center = Point(x+width/float(2), y+height/float(2))
        self.width  = width
        self.height = height


    def __str__(self):
        str=("(%4d,%4d)              (%4d,%4d)\n"
             "      .-----------------------.\n"
             "      |                       |\n"
             "      |                %6.1f |\n"
             "      |       %6.1f          |\n"
             "      '-----------------------'\n"
             "(%4d,%4d)              (%4d,%4d)"
             )
        nums=( self.l_top.x, self.l_top.y,         self.r_top.x, self.r_top.y,
                                                                         self.height,
                                         self.width,
               self.l_bot.x, self.l_bot.y,         self.r_bot.x, self.l_bot.y  )
        return str % nums


    def __iter__(self):
        yield self.l_top
        yield self.r_top
        yield self.r_bot
        yield self.l_bot


    # Gives back a copy of this rectangle
    def copy(self):
        return Rect(self.l_top.x, self.l_top.y, self.width, self.height)


    # Check to see if two croner points belong to the same edge
    def corners_belong_to_edge(self, c1, c2):
        return True in [
            (c1==self.l_top and c2==self.r_top) or
            (c1==self.r_top and c2==self.l_top) or
            (c1==self.r_top and c2==self.r_bot) or
            (c1==self.r_bot and c2==self.r_top) or
            (c1==self.r_bot and c2==self.l_bot) or
            (c1==self.l_bot and c2==self.r_bot) or
            (c1==self.l_bot and c2==self.l_top) or
            (c1==self.l_top and c2==self.l_bot) ]


    # ______
    #|    . |
    #|______|
    def is_point_inside_rect(self, point):
        return (self.l_top.x <= point.x <= self.r_top.x and
            self.l_top.y <= point.y <= self.l_bot.y)


    #  ______
    # |     _|____
    # |____|      |
    #      |______|
    def overlaps_with(self, rect):
        a = False
        for corner in rect:
            if self.is_point_inside_rect(corner):
                a = True
        for corner in self:
            if rect.is_point_inside_rect(corner):
                a = True
        return a

    #
    #
    #
    #
    def box_ovlp_word(self, w_rect):  # assuming rect is the word_rect and box/word are overlapping
        b = [self.l_top.x, self.l_top.y, self.r_bot.x, self.r_bot.y]  # [x1, y1, x2, y2]
        w = [w_rect.l_top.x, w_rect.l_top.y, w_rect.r_bot.x, w_rect.r_bot.y]  # [x1, y1, x2, y2]
        correct_overlapping = 1
        x_ovlp = max(0, min(w[2], b[2]) - max(w[0], b[0]))
        y_ovlp = max(0, min(w[3], b[3]) - max(w[1], b[1]))
        so = x_ovlp * y_ovlp  # surface of overlapping
        po = so / float(w_rect.width*w_rect.height)  # percent overlapping
        if self.l_top.y > w_rect.l_top.y or self.r_bot.y < w_rect.r_bot.y:
            correct_overlapping = 0
        return [x_ovlp, y_ovlp, po, correct_overlapping]

    #  ______                ____ ______
    # |     _|____          |    |      |
    # |____|      |   -->   |____|______|
    #      |______|
    def align_with_top_edge_of(self, rect):
        self.l_top.y = self.r_top.y = rect.r_top.y
        self.l_bot.y = self.r_bot.y = self.l_top.y+self.height
        return self


    #  ______                ______
    # |     _|____          |______|
    # |____|      |   -->   |      |
    #      |______|         |______|
    def align_with_left_edge_of(self, rect):
        self.l_top.x = self.l_bot.x = rect.l_top.x
        self.r_top.x = self.r_bot.x = self.l_top.x+self.width
        return self


    # ______
    #|      |
    #|______|
    #    ______
    #   |      |
    #   |______|
    def overlaps_on_x_axis_with(self, rect):
        return self.copy().align_with_top_edge_of(rect).overlaps_with(rect)


    # ______
    #|      |   ______
    #|______|  |      |
    #          |______|
    def overlaps_on_y_axis_with(self, rect):
        return self.copy().align_with_left_edge_of(rect).overlaps_with(rect)


    # ______
    #|      |             The calculation includes
    #|______|             both edges and corners.
    #        \ d
    #         \ ______
    #          |      |
    #          |______|
    def distance_to_rect(self, rect):

        # 1. see if they overlap
        if self.overlaps_with(rect):
            return 0

        # 2. draw line between rectangles
        line = (self.center, rect.center)
        #print "line=%s %s" % (line[0], line[1])

        # 3. find the two edges that intersect the line
        p1, p2 = None, None
        for corner in self:
            if corner.faces_line(line):
                if p1 is None:
                    p1=corner
                elif self.corners_belong_to_edge(corner, p1):
                    p2=corner
        edge1=(p1, p2)
        p1, p2 = None, None
        for corner in rect:
            if corner.faces_line(line):
                if p1 is None:
                    p1=corner
                elif rect.corners_belong_to_edge(corner, p1):
                    p2=corner
        edge2=(p1, p2)

        # 4. find shortest distance between these two edges
        distances=[
            distance_between_edge_and_point(edge1, edge2[0]),
            distance_between_edge_and_point(edge1, edge2[1]),
            distance_between_edge_and_point(edge2, edge1[0]),
            distance_between_edge_and_point(edge2, edge1[1]),
        ]

        return min(distances)



# ---------------------- Math primitive functions ----------------------

def distance_between_points(point1, point2):
    return point1.distance_to_point(point2)

def distance_between_rects(rect1, rect2):
    return rect1.distance_to_rect(rect2)

def triangle_area_at_points(p1, p2, p3):
    a=p1.distance_to_point(p2)
    b=p2.distance_to_point(p3)
    c=p1.distance_to_point(p3)
    s=(a+b+c)/float(2)
    area=sqrt(s*(s-a)*(s-b)*(s-c))
    return area

# Finds angle using cos law
def angle(a, b, c):
    divid=(a**2+b**2-c**2)
    divis=(2*a*b)
    if (divis)>0:
        result=float(divid)/divis
        if result<=1.0 and result>=-1.0:
            return acos(result)
        return 0
    else:
        return 0

# Checks if point faces edge
def point_faces_edge(edge, point):
    a=edge[0].distance_to_point(edge[1])
    b=edge[0].distance_to_point(point)
    c=edge[1].distance_to_point(point)
    ang1, ang2 = angle(b, a, c), angle(c, a, b)
    if ang1>pi/2 or ang2>pi/2:
        return False
    return True

# Gives distance if the point is facing edge, else False
def distance_between_edge_and_point(edge, point): # edge is a tupple of points
    if point_faces_edge(edge, point):
        area=triangle_area_at_points(edge[0], edge[1], point)
        base=edge[0].distance_to_point(edge[1])
        height=area/(0.5*base)
        return height
    return min(distance_between_points(edge[0], point),
               distance_between_points(edge[1], point))





# TEST #

def test0():
    text_in_file = {'im0.jpg': [['lala', 0, 0, 2, 2]]}
    ref_segmentation = {'im0.jpg': {'0': [0, 5, 0, 5]}}
    [a, b] = associate_text_product(text_in_file, ref_segmentation, "")
    assert a == {'im0.jpg': {'0': [0, 0, 5, 5, ['lala'], [[0, 0, 2, 2]], [0]]}}
    assert b == {'im0.jpg': {0: ['lala', 0, 0, 2, 2, ['0', 4]]}}
    return "Test0 passed"

if __name__ == '__main__':
    print(test0())
