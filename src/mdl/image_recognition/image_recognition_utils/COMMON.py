import os

##################################################
# !!! TO BE CHANGED DEPENDING ON YOUR SYSTEM !!! #
caffe_root='/home/ubuntu/caffe/'
storage_path=os.environ['Q_PATH']+'/QOPIUS_STORAGE/'


##################################################
# NO CHANGE NEEDED
#path_to_utils=os.path.dirname(os.path.realpath(__file__)) + '/../utils/'
path_to_utils = os.path.dirname(os.path.realpath(__file__))
path_to_sheets = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/general_database/'
net_root_path = storage_path + 'saved_models/'
data_root_path = storage_path + 'Picture_database/'
general_database_storage_path = data_root_path + 'general_database_storage/'
path_to_getters = os.path.realpath(__file__)+'/'  # KEVIN 04102016

qopiusVisual = os.environ['Q_PATH'] + '/q-engine/qopius_visual'
textRecognition = qopiusVisual + '/text_recognition'

##################################################
# Tensorflow paths
tensorflow_root_im_database_path = data_root_path + 'im_database/'
tensorflow_root_im_database_reference_path = data_root_path + 'im_database_reference/'
tensorflow_root_bottleneck_database_path = storage_path + 'bottleneck_database/'
tensorflow_saved_models_path = net_root_path + 'tensorflow_models/'
tensorflow_inception_model_path = tensorflow_saved_models_path + 'inception/'
tensorflow_utils = path_to_utils + '/utils_tensorflow/'

##################################################
# Tensorbox paths
tensorbox_saved_models_path = net_root_path + '/tensorbox_models/'
tensorbox_utils = path_to_utils + '/utils_tensorbox'
trainTensorbox = path_to_utils + '/utils_tensorbox/train'

siamese_saved_models_path = net_root_path + 'siamese_models/'
facenet_saved_models_path = net_root_path + 'facenet_models/'

##
## { item_description }
##

storage = storage_path
imageStorage = data_root_path
imCanvas = tensorflow_root_im_database_path
imReference = tensorflow_root_im_database_reference_path
imShelf = data_root_path + 'im_database_shelf/'
botCanvas = tensorflow_root_bottleneck_database_path + 'bottleneck_im_database'
botReference = tensorflow_root_bottleneck_database_path + 'bottleneck_im_database_reference'
inceptionV3 = tensorflow_saved_models_path + 'inception/'
tensorflowModels = tensorflow_saved_models_path

botDatabase = tensorflow_root_bottleneck_database_path
dictionaries = data_root_path + 'dictionaries'
