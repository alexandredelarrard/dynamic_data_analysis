import tensorflow as tf
import sys

# change this as you see fit
#image_path = sys.argv[1]

# Read in the image_data
#image_data = tf.gfile.FastGFile(image_path, 'rb').read()
import os
import shutil
import time
import sys
from os import listdir
from os import mkdir
from shutil import copyfile
from os.path import isfile, join
from PIL import Image
import numpy as np
d = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # /home/kristina/desire-directory
sys.path.append(d)
import COMMON
from multiprocessing.pool import ThreadPool

class TestTensorflow(object):
    def __init__(self, params, config_tensorflow, n_best):
        # Initialize tensorflow session and check if (family, user_status)  is in list of parameters
        self.params = params
        self.config_tensorflow = config_tensorflow

        assert 'family' in self.params
        assert 'user_status' in self.params

        self.nb_top_brand = n_best
        self.nb_top_upc = n_best

        #### Try to shut logging to print only errors -> not really successful
        self.params['verbose'] = False

        ################# --- DATA PATHS --- #############
        assert 'path_to_data' in self.params
        self.varPath = self.params['path_to_data']
        if self.varPath != None :
            self.imgFiles = [f for f in listdir(self.varPath) if isfile(join(self.varPath, f))]

        print self.varPath

        ################# --- MODEL PATHS --- #############
        if self.params['user_status'] == 'admin':
            root_saved_models = COMMON.tensorflow_saved_models_path
        else:
            root_saved_models = self.params['path_to_data']+'/saved_models/tensorflow_models'
        root_model = root_saved_models+'/'+self.params['family']

        # if a 'brand_id' was given, we are looking for specific UPC models
        if 'brand_id' in self.params:
            root_model += '/'+self.params['brand_id']

        # get model files
        self.output_graph = root_model+'/output_graph.pb'      # Where to look for the trained graph.
        self.output_labels = root_model+'/output_labels.txt'   # Where to look for the trained graph\'s labels.


        ################# --- CHECK EXISTENCE OF MODEL PATHS --- ###############

        self.brand_model = False
        if 'brand_id' not in self.params:
            # only looking for family model
            if not os.path.exists(self.output_labels) or not os.path.exists(self.output_graph):
                # models don't exist in clients saved models
                # go check for QOPIUS_STORAGE saved models
                self.output_graph = COMMON.tensorflow_saved_models_path + '/' + self.params['family'] + '/output_graph.pb'
                self.output_labels = COMMON.tensorflow_saved_models_path + '/' + self.params['family'] + '/output_labels.txt'

                # check these exist - stop program if files not found
                if os.path.exists(self.output_labels) and os.path.exists(self.output_graph):
                    print("- [STATUS: OK] - output_labels found at: {}".format(self.output_labels))
                    print("- [STATUS: OK] - output_graph found at: {}".format(self.output_graph))
                else:
                    raise Exception("FATAL ERROR: 'output_labels.txt' / 'output_graph.pb' not found -> tensorflow cannot proceed")

            else:
                print("- [STATUS: OK] - output_labels found at: {}".format(self.output_labels))
                print("- [STATUS: OK] - output_graph found at: {}".format(self.output_graph))

        else:
            # looking for brand models
            if os.path.exists(self.output_labels) and os.path.exists(self.output_graph):
                print("- [STATUS: OK] - output_labels found at: {}".format(self.output_labels))
                print("- [STATUS: OK] - output_graph found at: {}".format(self.output_graph))

                self.brand_model = True

            else:
                # models don't exist in clients saved models
                # go check for QOPIUS_STORAGE saved models
                self.output_graph = COMMON.tensorflow_saved_models_path + '/' + self.params['family'] + '/' + self.params['brand_id'] + '/output_graph.pb'
                self.output_labels = COMMON.tensorflow_saved_models_path + '/' + self.params['family'] + '/' +self.params['brand_id'] + '/output_labels.txt'

                # if models found (it's possible they are not found, since they are not mandatory...)
                if os.path.exists(self.output_labels) and os.path.exists(self.output_graph):
                    print("- [STATUS: OK] - output_labels found at: {}".format(self.output_labels))
                    print("- [STATUS: OK] - output_graph found at: {}".format(self.output_graph))
                    self.brand_model = True

                else:
                    print("- [STATUS: FAIL] - output_labels NOT found at: {}".format(self.output_labels))
                    print("- [STATUS: FAIL] - output_graph NOT found at: {}".format(self.output_graph))


    def _performance_metrics_single(self, imageFile, top_k, label_lines, predictions):
        im_label = imageFile.split('_', 1)[0]
        rank = 1
        for node_id in top_k:
            if label_lines[node_id] == im_label:
                score = predictions[0][node_id]
                return [rank, score, True]
            else:
                rank += 1
        return [rank, 0, False]

    def _performance_metrics_global(self, pm):
        n = len(pm)
        top_1 = 0
        top_5 = 0
        top_10 = 0
        outclass = 0

        for im, pms in pm.items():
            if pms[2]:
                if pms[0] <= 10:
                    top_10 += 1
                if pms[0] <= 5:
                    top_5 += 1
                if pms[0] <= 1:
                    top_1 += 1
            else:
                outclass += 1

        if n-outclass > 0:
            top_1 /= float(n-outclass)
            top_5 /= float(n-outclass)
            top_10 /= float(n-outclass)

        if self.params['verbose']:
            print("\n= Detail of performance per image =")
            for k in pm:
                print("- %25s -> rank: %2d | score: %4.2f | outclass?: %s" % (k, pm[k][0], pm[k][1], str(pm[k][2])))

        print("\n\n= Total performance on %d images | outclass images: %d | remaining images: %d =" % (n, outclass, n-outclass))
        print("- Absolute -> %.2f" % top_1)
        print("- First 5  -> %.2f" % top_5)
        print("- First 10 -> %.2f" % top_10)

    def _progress_bar(self, i, total):
        if total > 20:
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*(i*20/total), 100*i/total))
            sys.stdout.flush()

    def main(self):

        print("\n\n\n========================\n==== TEST ====\n========================\n")
        # Loads label file, strips off carriage return
        label_lines = [line.rstrip() for line
                           in tf.gfile.GFile(self.output_labels)]

        # Unpersists graph from file
        print("=====================    Unpersist graph   =======================")
        start_0 = time.time()
        with tf.gfile.FastGFile(self.output_graph, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        end_0 = time.time()

        print("= Inject images =")
        start_1 = time.time()

        with tf.Session(config = self.config_tensorflow) as sess:
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

            pm = {}  # performance metrics
            i = 0
            total = len(self.imgFiles)
            print("\n \n Progress bar")
            for imageFile in self.imgFiles:

                image_data =  tf.gfile.FastGFile(self.varPath+"/"+imageFile, 'rb').read()

                if self.params['verbose']: print (self.varPath+"/"+imageFile)

                predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

                # Sort to show labels of first prediction in order of confidence
                top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                firstElt = top_k[0];

                newFileName = label_lines[firstElt] +"--"+ str(predictions[0][firstElt])[2:7]+".jpg"
                if self.params['verbose']: print(newFileName)

                # performance metrics
                single_pm = self._performance_metrics_single(imageFile, top_k, label_lines, predictions)  # single_pm = [rank, confidence, (BOOLEAN) class_had_been_trained] (rank ranges from 1 to XX)
                pm[imageFile] = single_pm

                for node_id in top_k:
                    human_string = label_lines[node_id]
                    score = predictions[0][node_id]
                    #print (node_id)
                    if self.params['verbose']: print('%s (score = %.5f)' % (human_string, score))

            end_1 = time.time()

            # print global performance metric
            self._performance_metrics_global(pm)

            # print processing time recap
            print("\n\n= Time recap =")
            print("- time to unpersist graph from file: %.3f seconds" % (end_0 - start_0))
            print("- time to process test directory: %.3f seconds" % (end_1 - start_1))
            print("- total time: %.3f seconds | time per image: %.3f seconds" % ((end_1 - start_0), (end_1 - start_0)/total))

    def Run(self, sess, softmax, image, image_number):
        prediction = sess.run(softmax, {'DecodeJpeg:0': image})
        return prediction

    def brand_wrapper(self, client, api_key,  object_dict, image_dict):

        # Loads label file, strips off carriage return
        label_lines = [line.rstrip() for line in tf.gfile.GFile(self.output_labels)]
        path_to_output_image = '/home/ubuntu/clients_storage/' + client + "/" + api_key + "/temp/out_image/"
        pool = ThreadPool()

        # Unpersists graph from file
        # print("=========== Unpersist graph brand wrapper ===========")
        start_0 = time.time()
        with tf.gfile.FastGFile(self.output_graph, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        end_0 = time.time()

        # print("=========== Inject images in brand wrapper ========")
        start_1 = time.time()
        self.config_tensorflow.intra_op_parallelism_threads=80
        
        with tf.Session(config=self.config_tensorflow) as sess:
            # Feed the image_data as input to the graph and get first prediction
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

            total = len(image_dict)
            out_object_dict={}

            for imageFile in image_dict:

                print 'image : %s under process' %imageFile
                out_object_dict[imageFile]={}
                image_data =[]

                # create all crops of each picture and stock it to image_data
                
                for idbox in object_dict[imageFile]:
                    box_info = object_dict[imageFile][idbox]

                    x1 = box_info[2]
                    x2 = box_info[3]
                    y1 = box_info[0]
                    y2 = box_info[1]

                    # feed to the network
                    image_data.append(image_dict[imageFile].crop((x1, y1, x2, y2)))

                # Multithread version of brands
                threads = [pool.apply_async(self.Run, args=(sess, softmax_tensor, image_data[image_number], image_number)) for
                           image_number in range(len(image_data))]

                for idbox, thread in enumerate(threads):
                    result = thread.get()

                    # Sort to show labels of first prediction in order of confidence
                    top_k = result[0].argsort()[-len(result[0]):][::-1]
                    firstElt = top_k[0];
                    br = []
                    sc = []
                    for node_id in top_k[0:self.nb_top_brand]:
                        br.append(int(label_lines[node_id].split('_')[0]))
                        sc.append(result[0][node_id])

                    out_object_dict[imageFile][idbox] = list(object_dict[imageFile][idbox][0:4])
                    out_object_dict[imageFile][idbox].append(np.array(br))
                    out_object_dict[imageFile][idbox].append(np.array(sc))

                    if self.params['verbose']: print('- Image: %s | Box: %3d | Brands: %30s |  Score_first: %.5f' % (imageFile, idbox, str(br), sc[0]))

            end_1 = time.time()
            # print processing time recap
            # print("\n\n= Time recap TEST TENSORFLOW =")
            # print("- time to unpersist graph from file: %.3f seconds" % (end_0 - start_0))
            # print("- time to process test directory: %.3f seconds" % (end_1 - start_1))
            # if total>0 :
                # print("- total time: %.3f seconds | time per image: %.3f seconds" % ((end_1 - start_0), (end_1 - start_0)/total))

        return out_object_dict

    def upc_wrapper(self, client, api_key, object_dict, image_dict):

        path_to_output_image = '/home/ubuntu/clients_storage/' + client + "/" + api_key + "/temp/out_image/"

        objects = []
        out_object_dict={}
        tf.reset_default_graph()

        if self.brand_model:
            # Loads label file, strips off carriage return
            label_lines = [line.rstrip() for line in tf.gfile.GFile(self.output_labels)]

            # Unpersists graph from file
            # print("=========== Unpersist graph upc wrapper ===========")
            start_0 = time.time()
            with tf.gfile.FastGFile(self.output_graph, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='')
            end_0 = time.time()

            # print("===========   Inject images in upc wrapper ========")
            start_1 = time.time()

            with tf.Session(config=self.config_tensorflow) as sess:
                # Feed the image_data as input to the graph and get first prediction
                softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

                for box_info in object_dict:
                    imageFile = box_info[4]
                    x1 = box_info[2]
                    x2 = box_info[3]
                    y1 = box_info[0]
                    y2 = box_info[1]
                    idBox=str(box_info[5]) + '_' + imageFile

                    # feed to the network
                    image_data = image_dict[imageFile].crop((x1, y1, x2, y2))
                    predictions = sess.run(softmax_tensor, {'DecodeJpeg:0': image_data})

                    # Sort to show labels of first prediction in order of confidence
                    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                    firstElt = top_k[0];

                    upc = []
                    sc = []
                    for node_id in top_k[0:self.nb_top_upc]:
                        upc.append(int(label_lines[node_id]))
                        sc.append(predictions[0][node_id])

                    out_object_dict[idBox]=box_info
                    out_object_dict[idBox].append(np.array(upc))
                    out_object_dict[idBox].append(np.array(sc))

                    if self.params['verbose']: print('- Image: %s | Box: %3d | UPC: %30s |  Score_first: %.5f' % (imageFile, idbox, str(upc), sc[0]))

                end_1 = time.time()
                # print("\n\n= Time recap TEST TENSORFLOW =")
                # print("- time to unpersist graph from file: %.3f seconds" % (end_0 - start_0))
                # print("- time to process test directory: %.3f seconds" % (end_1 - start_1))
                # print("- total time: %.3f seconds | time per image: %.3f seconds" % ((end_1 - start_0), (end_1 - start_0)/len(image_dict)))

        else:
            print("No model found for brand_id: %s" % self.params['brand_id'])
            for box in object_dict:
                imageFile = box[4]
                idBox=str(box[5]) + '_' + imageFile

                box += [np.array([-1 for i in range(self.nb_top_upc)]), np.array([-1 for i in range(self.nb_top_upc)])]
                out_object_dict[idBox]=box

        return out_object_dict
