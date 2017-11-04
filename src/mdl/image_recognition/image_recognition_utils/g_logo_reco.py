#!/usr/bin/env python


import base64
from cStringIO import StringIO
import os
import re
from googleapiclient import discovery
from googleapiclient import errors
from oauth2client.client import GoogleCredentials
from PIL import Image
from PIL import ImageDraw
import numpy as np

BATCH_SIZE = 10 

root_path = os.path.abspath(os.path.join(__file__, '..','..'))

class VisionApi:
    """Construct and use the Google Vision API service."""

    DISCOVERY_URL = 'https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'  # noqa
    
    #os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/home/bertin/cloud-vision/python/qtext-bbd92248f46b.json' #WARNING: Change with pc
    
    
    def __init__(self,api_key_path, api_discovery_file='vision_api.json'):

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =api_key_path 
        
        self.credentials = GoogleCredentials.get_application_default()
        self.service = discovery.build(
            'vision', 'v1', credentials=self.credentials,
            discoveryServiceUrl='https://{api}.googleapis.com/$discovery/rest?version={apiVersion}')

    def detect_logo(self, reference_data_croped, num_retries=3, max_results=100):
        """Uses the Vision API to detect text in the given file.
        """

        batch_request = []
        for filename in reference_data_croped:

            output = StringIO()
            reference_data_croped[filename].save(output, format='JPEG')
            im = output.getvalue()

            batch_request.append({
                'image': {
                    'content': base64.b64encode(im
                            ).decode('UTF-8')
                },
                'features': [{
                    'type': 'LOGO_DETECTION',
                    'maxResults': max_results,
                    
                }]
            })
        request = self.service.images().annotate(
            body={'requests': batch_request})
        print('ready to send to Logo API ')
        try:
            responses = request.execute(num_retries=num_retries)
            print(responses)
            if 'responses' not in responses:
                return {}
            text_response = {}
            for filename, response in zip(reference_data_croped.keys(), responses['responses']):
                if 'error' in response:
                    print("API Error for %s: %s" % (
                            filename,
                            response['error']['message']
                            if 'message' in response['error']
                            else ''))
                    continue
                if 'logoAnnotations' in response:
                    text_response[filename] = response['logoAnnotations']
                else:
                    text_response[filename] = []
            return text_response
        except errors.HttpError as e:
            print("Http Error for %s: %s" % (filename, e))
        except KeyError as e2:
            print("Key error: %s" % e2)
# [END detect_text]


def get_logo_from_files(vision,reference_data_croped):
    """Call the Vision API on a file and index the results."""


    
    texts = vision.detect_logo(reference_data_croped)
 
    
    return texts


def batch(iterable, batch_size=BATCH_SIZE):
    """Group an iterable into batches of size batch_size.

    >>> tuple(batch([1, 2, 3, 4, 5], batch_size=2))
    ((1, 2), (3, 4), (5))
    """
    b = []
    for i in iterable:
        b.append(i)
        if len(b) == batch_size:
            yield tuple(b)
            b = []
    if b:
        yield tuple(b)

def highlight_text(input_image,texts,output_filename):

    input_image.save('prov.jpg', format='JPEG')
    
    with open('prov.jpg') as image:

        # Reset the file pointer, so we can read the file again
        image.seek(0) 
 
        im = Image.open(image)
        draw = ImageDraw.Draw(im)
        
        
        u=0
        for face in texts:
         

                descrip = face['description']

                box = [(v.get('x', 0.0), v.get('y', 0.0)) for v in face['boundingPoly']['vertices']]

                draw.line(box + [box[0]], width=5, fill='#00ff00')

             
            
        del draw
        im.save(output_filename)
        
def get_logo(image_name,texts,text_found):

        
    u=0
   
    for face in texts:
        
 
        #if u>0:
            out_res=[image_name]

            descrip = face['description']

            out_res.append(re.sub(r'\W+', '', descrip.encode('ascii', 'ignore')))

            box = [(v.get('x', 0.0), v.get('y', 0.0)) for v in face['boundingPoly']['vertices']]
   
            if (box[1][0]-box[0][0])>0 and (box[2][1]-box[1][1])>0 and out_res[1]!='' and out_res[1].isdigit()==False:
            
                
                min_X=np.min([box[0][0],box[1][0],box[2][0],box[3][0]])
                min_Y=np.min([box[0][1],box[1][1],box[2][1],box[3][1]])
                max_X=np.max([box[0][0],box[1][0],box[2][0],box[3][0]])
                max_Y=np.max([box[0][1],box[1][1],box[2][1],box[3][1]])

                out_text=[]
                out_text.append([min_X,min_Y])
                out_text.append([max_X,min_Y])
                out_text.append([max_X,max_Y])
                out_text.append([min_X,max_Y])
  

                out_res.append(out_text)
                text_found.append(out_res)

        
        #u=+1
    
    return text_found
 
# function to apply text recognition on images with google vision api
# return an array of the text recognized in each crop of each image

           
def g_logo_reco(ref_crop,reference_data_croped,api_key_path, queue=[],display_results=False):

    ### INPUT
    # ref crop is a dictionary :
    #key : croped image name
    # values =[initial image name, Y start, X start ]of croped image
    ### reference_data_croped is a dictionary of croped images
    ## key : croped image file name
    ## values : croped images  

    text_found=[]
    
    # Create a client object for the Vision API
    vision = VisionApi(api_key_path)
                                
    texts=get_logo_from_files(vision,reference_data_croped)

    for text in texts:
        

        
        text_found=get_logo('%s' %text,texts['%s' %text],text_found)
        
        
        
        out_path_im='%s_results.jpg' %text
        if display_results==True:
            highlight_text(reference_data_croped['%s' %text],texts['%s' %text],out_path_im)
        
    #queue.put(text_found)       
    return text_found       
    
