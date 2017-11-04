# imports
import numpy as np
import collections
import os
import cv2
os.environ['GLOG_minloglevel'] = '2'
import pandas as pd
from highlight_boxes import * # contains highlight_boxes_price
from PIL import Image, ImageDraw,ImageFont
from scipy.misc import toimage
## ALL PLOT FUNCTIONS USED BY IMAGE_TO_UPC

# just plot images
def plot_image_simple(path_images_out, reference_data, suffix=0):
    for image_origin_name, im in reference_data.items():
        im.save(path_images_out+'/{}_{}.{}'.format(image_origin_name.split('.', 1)[0], suffix, image_origin_name.split('.', 1)[1]))


## draw product boxes
def plot_segmentation(ref_segmentation,path_images_out,reference_data):
# ref_segmentation = {image_origin_name: {idbox: [y1, y2, x1, x2]}}

    for image_origin_name, boxes_dict in ref_segmentation.items():
        out_path_im = path_images_out + '/%s_results_seg.jpg' % image_origin_name
        in_boxes=[]
        img = np.array(toimage(reference_data[image_origin_name]))
        for id_box, boxes in boxes_dict.items():
            ### numpy image, (x1,y1) , (x2, y2) , (color), thickness
            cv2.rectangle(img, (boxes[2],boxes[0]), (boxes[3],boxes[1]), (0,255,0), 3)

        cv2.imwrite(out_path_im, img)

###
# draw price boxes
def plot_segmentation_price(ref_segmentation, path_images_out, reference_data):

    for image_origin_name, boxes_dict in ref_segmentation.items():
        out_path_im = path_images_out + '/%s_results_etiquettes.jpg' % image_origin_name
        in_boxes=[]
        img = np.array(toimage(reference_data[image_origin_name]))
        for id_box, boxes in boxes_dict.items():
            ### [x1, y1, x2 , y2]
            cv2.rectangle(img, (boxes[2],boxes[0]), (boxes[3],boxes[1]), (0,255,0), 3)

        cv2.imwrite(out_path_im, img)

###
# draw squared boxes of prices initially cropped
###
def plot_relative_ref_segmentation(ref_segmentation, path_images_out, reference_data):

    for image_origin_name, boxes_dict in ref_segmentation.items():
        out_path_im = path_images_out + '/%s_squared_prices.jpg' % image_origin_name
        in_boxes=[]
        img = np.array(toimage(reference_data[image_origin_name]))
        for id_box, boxes in boxes_dict.items():
            ### [x1, y1, x2 , y2]
            cv2.rectangle(img, (boxes[2],boxes[0]), (boxes[3],boxes[1]), (0,255,0), 3)

        cv2.imwrite(out_path_im, img)

###
# draw global images composed of all boxes of price resized as square
###
def plot_image_concatenation(image_concatenation,path_images_out):
#image_concatenation = {image_origin_name: [price_image, {idbox: [y1, y2, x1, x2]}}

    for image_origin_name, im_info in image_concatenation.items():

        box_all=[]
        if len(im_info)>0 :

            for boxId,box in im_info[1].items():
                box_all.append(im_info[1][boxId])

            out_path_im = path_images_out + '/%s_concat_prices.jpg' % image_origin_name
            draw_boxes(im_info[0], box_all, out_path_im)

###
# draw bound product and bound price (corresponding to shelf identification)
###
def draw_bound(shelf_bound_list_price,shelf_bound_list_product,path_images_out, reference_data,all_out_of_stock,ref_segmentation_product):

    for image_origin_name in reference_data:

        out_path_im = path_images_out + '/%s_results_bounds.jpg' % image_origin_name

        path_temp= path_images_out +'/temp.jpg'
        reference_data[image_origin_name].save(path_temp, format='JPEG')
        with open(path_temp) as image:
           # Reset the file pointer, so we can read the file again
           image.seek(0)
           im = Image.open(image)
           draw = ImageDraw.Draw(im)

           for u in shelf_bound_list_price[image_origin_name]:
              draw.line((0,u,im.size[0], u), fill=18, width=7)

           for u in shelf_bound_list_product[image_origin_name]:
              draw.line((0,u,im.size[0], u), fill=128, width=7)

           ## print boxes
           if image_origin_name in ref_segmentation_product :
               
               seg_info = ref_segmentation_product[image_origin_name]
               
               for box in seg_info : 
                   
                   #draw.rectangle([box['x1']  ,box['y1']  ,box['x2'] ,box['y2']], outline='#00ff00')
                   
                   draw_rectangle(draw,[[ seg_info[box][2] , seg_info[box][0]]  ,[ seg_info[box][3] , seg_info[box][1] ]], "black", width=5)               

           
           ## print out of stock boxes
           if image_origin_name in all_out_of_stock :
               
               out_of_stock = all_out_of_stock[image_origin_name]
               for box in out_of_stock : 
                   
                   #draw.rectangle([box['x1']  ,box['y1']  ,box['x2'] ,box['y2']], outline='#00ff00')
                   draw_rectangle(draw,[[box['x1']  ,box['y1']]  ,[box['x2'] ,box['y2']]], "blue", width=5)
                   
           im.save(out_path_im)                   
           del draw                   


#
## Draw recognized_price (output of g_text_reco)
##
def plot_recognized_text(recognized_price, path_images_out, add_filename, reference_data_price):
## recognized_price = [[image_origin_name, price_word, [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]], ...]  (points are in indirect sense from top left)
## reference_data_price {image_origin_name: price_image}

    for image_origin_name in reference_data_price:

        out_path_im = path_images_out + '/%s_%s.jpg' % (image_origin_name, add_filename)
        text_in=[]

        if recognized_price !=[] :
            for box_text in recognized_price :

                if box_text!=[]:
                    if box_text[0]==image_origin_name:
                       text_in.append([box_text[1],box_text[2]])
            draw_text(reference_data_price[image_origin_name], text_in, out_path_im)

## Draw prices or text in global image
#
def plot_text_global(text_in_file,path_images_out,add_filename,reference_data):
    # text_in_file = {filename:[[word, X1, Y1, X2, Y2]]}

    for image_origin_name, boxes_dict in text_in_file.items():
        out_path_im = path_images_out + '/%s_%s.jpg' %(image_origin_name,add_filename)
        text_in=[]

        for box_text in boxes_dict :
            if box_text!=[]:
               text_in.append([box_text[2],box_text[4],box_text[1],box_text[3],box_text[0]])

        draw_boxes_and_text(reference_data[image_origin_name], text_in, out_path_im)

## Draw prices associated to each products
#
def plot_pr_product(price_product, path_images_out, reference_data):

   for im_file in price_product :

        out_path_im=path_images_out +'/%s_pr_product.jpg' %im_file
        highlight_boxes_price(reference_data[im_file],price_product[im_file],out_path_im)

def draw_boxes(input_image, boxes, output_filename):

    width =5
    draw = ImageDraw.Draw(input_image)

    for box_b in boxes:
        ### [x1, y1, x2 , y2]
        cor = [int(box_b[2]),int(box_b[0]),int(box_b[3]),int(box_b[1])]

        for i in range(width):
            draw.rectangle(cor, outline='#00ff00')
            cor = (cor[0]+1,cor[1]+1, cor[2]+1,cor[3]+1)

    del draw
    toimage(input_image).save(output_filename)


def draw_text(input_image, list_text, output_filename):

    draw = ImageDraw.Draw(input_image)

    for box_text in list_text:
        draw.text((box_text[1][0][0],box_text[1][1][1]),str(box_text[0]))
    toimage(input_image).save(output_filename)
    del draw


def highlight_boxes_price(input_image,boxes,output_filename):
    draw = ImageDraw.Draw(toimage(input_image))
    yy=0

    for box_b in boxes:
        yy=yy+1
        if boxes[box_b] != [] :
            box = [(boxes[box_b][2],boxes[box_b][0]),(boxes[box_b][3],boxes[box_b][0]),(boxes[box_b][3],boxes[box_b][1]),(boxes[box_b][2],boxes[box_b][1])]

            draw.line(box + [box[0]], width=5, fill='#00ff00')

    toimage(input_image).save(output_filename)
    del draw


def draw_boxes_and_text(input_image, boxes, output_filename) :

    path_temp= os.path.dirname(output_filename) +'/temp.jpg'
    input_image.save(path_temp, format='JPEG')
    width = 5

    with open(path_temp) as image:

        # Reset the file pointer, so we can read the file again
        image.seek(0)
        im = Image.open(image)
        draw = ImageDraw.Draw(im)

        u=0
        for box_b in boxes:

            draw.text((int(box_b[2])+15,int(box_b[0])+15),str(box_b[4]))
            cor = [int(box_b[2]),int(box_b[0]),int(box_b[3]),int(box_b[1])]

            for i in range(width):
                draw.rectangle(cor, outline='#00ff00')
                cor = (cor[0]+1,cor[1]+1, cor[2]+1,cor[3]+1)

        im.save(output_filename)
        del draw

def draw_rectangle(draw,coordinates, color, width=1):
    
    for i in range(width):
        rect_start = (coordinates[0][0] - i, coordinates[0][1] - i)
        rect_end = (coordinates[1][0] + i, coordinates[1][1] + i)
        draw.rectangle((rect_start, rect_end), outline = color)