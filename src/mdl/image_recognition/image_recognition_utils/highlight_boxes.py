from PIL import Image
from PIL import ImageDraw
import numpy as np
## Antonin ###

## Function to highlight boxes of text or logo
## INPUT : 
### input_image : image file name
### texts : text or logo recognized in json format
### output_filenmae : ouptut file name


def highlight_boxes(input_image,texts,output_filename):


    input_image.save('/home/ubuntu/q-engine/qopius_visual/out_images/prov.jpg', format='JPEG')
    
    with open('/home/ubuntu/q-engine/qopius_visual/out_images/prov.jpg') as image:

        # Reset the file pointer, so we can read the file again
        image.seek(0) 
 
        im = Image.open(image)
        draw = ImageDraw.Draw(im)        
        
        u=0
        for box_b in texts:
        

                box = [(box_b[3],box_b[1]),(box_b[4],box_b[1]),(box_b[4],box_b[2]),(box_b[3],box_b[2])]

                draw.line(box + [box[0]], width=5, fill='#00ff00')

             
            
        del draw
        im.save(output_filename)


'''
def highlight_boxes_price(input_image,boxes,output_filename):

    input_image.save('/home/ubuntu/q-engine/qopius_visual/out_images/prov.jpg', format='JPEG')
    
    with open('/home/ubuntu/q-engine/qopius_visual/out_images/prov.jpg') as image:

        # Reset the file pointer, so we can read the file again
        image.seek(0) 
 
        im = Image.open(image)
        draw = ImageDraw.Draw(im)
        for box_b in boxes:
               
                box = [(boxes[box_b][2],boxes[box_b][0]),(boxes[box_b][3],boxes[box_b][0]),(boxes[box_b][3],boxes[box_b][1]),(boxes[box_b][2],boxes[box_b][1])]

                draw.line(box + [box[0]], width=5, fill='#00ff00')
				
        del draw
        im.save(output_filename)
'''
