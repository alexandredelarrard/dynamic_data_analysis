# -*- coding: utf-8 -*-"""Created on Mon Nov 27 09:01:10 2017
@author: ywxl"""
import numpy as npimport pandas as pdimport osimport jsonimport globfrom PIL import Image
from keras.models import Sequentialfrom keras.layers import Dense, Dropout,  Activation,Flattenfrom keras.layers import Conv2D, MaxPooling2Dfrom keras.preprocessing.image import ImageDataGeneratorfrom keras.callbacks import EarlyStopping, ModelCheckpointfrom keras import optimizersimport matplotlib.pyplot as pltfrom keras.regularizers import l2from keras import backend as K
from scipy.misc import imresize
class Train_keras(object):
    def __init__(self):
        self.batch_size         = 16        self.image_size         = 128        self.nb_epoch           = 300        self.keep_probability   = 0.6        self.weight_decay       = 0.000        self.learning_rate      = 0.0003        self.epoch_size         = 1000        self.patience           = 20
        self.seed               = 7666        self.data_augmentation  = True        self.pretrained_model   = None        self.proportion_train_test = 0.2
        self.path_client        =  r"U:\Technical\Actuarial Research\0330_Image Recognition\data\crashes_no_crashes"#r"U:\Technical\Actuarial Research\0330_Image Recognition\data\Cars"        self.Main()
    def Main(self):
        im_list_path = glob.glob(self.path_client + "\\frontcrash\*")        im_list_path += glob.glob(self.path_client + "\\no_crash\\Google - Cars\\*")                im_list_path= [x for x in im_list_path if "Thumbs.db" not in x]                liste = pd.DataFrame(im_list_path)[0].apply(lambda x: os.path.dirname(x).replace(self.path_client + "\\", ""))
        label_map = {}        for j, classe in enumerate(liste.unique()):            label_map[classe] = j
        x_train, y_train = self.load_data(im_list_path, label_map)
        weights= pd.DataFrame(np.argmax(y_train, 1))[0].value_counts()        dict_weights = {}        max_it       = weights[weights.index.tolist()[0]]        for i in weights.index.tolist():            dict_weights[i] = max_it/float(weights[i])
        #### train model keras 5 conv + 3 dense        self.model = self.Keras_model(y_train, x_train, self.image_size, label_map, dict_weights)
        #serialize model to JSON        model_json = self.model.to_json()        with open(self.path_client + "/model.json", "w") as json_file:            json_file.write(model_json)
        with open(self.path_client + "/classes.json", "w") as json_file:            json.dump(label_map, json_file)
    def setup_generator(self, X_train, X_valid, Y_train, Y_valid, batch_size=32):
        train_datagen = ImageDataGenerator(                                    shear_range=0.1,                                    zoom_range=0.3,                                    rotation_range=5,                                    width_shift_range=0.4,                                    height_shift_range=0.4,                                    zca_whitening=False,                                    fill_mode='reflect')
        train_datagen.fit(X_train)
        test_datagen = ImageDataGenerator(zoom_range=0.01)        test_datagen.fit(X_valid)
        X_train_aug = train_datagen.flow(X_train, Y_train, seed=self.seed, batch_size=self.batch_size)        X_test_aug = test_datagen.flow(X_valid, Y_valid, seed=self.seed, batch_size=self.batch_size)
        return X_train_aug, X_test_aug
    def Keras_model(self, y_train, x_train, image_resize, labels, dict_weights):
        X_rand = np.random.uniform(0,1,len(x_train))        test_index  = X_rand <= self.proportion_train_test        train_index = X_rand  >  self.proportion_train_test
        X_train = x_train[train_index]        Y_train = y_train[train_index]        X_valid = x_train[test_index]        Y_valid = y_train[test_index]
        print('Split train: ', len(X_train), len(Y_train))        print('Split valid: ', len(X_valid), len(Y_valid))
        kfold_weights_path = self.path_client + "/weights_kfold.h5"                model = Sequential()        model.add(Conv2D(32, (3, 3), padding='same',                         input_shape=(image_resize, image_resize, 3)))                model.add(Activation('relu'))        model.add(Conv2D(32, (3, 3)))        model.add(Activation('relu'))        model.add(MaxPooling2D(pool_size=(2, 2)))#        model.add(Dropout(0.25))                model.add(Conv2D(64, (3, 3), padding='same'))        model.add(Activation('relu'))                model.add(Conv2D(64, (3, 3)))        model.add(Activation('relu'))        model.add(MaxPooling2D(pool_size=(2, 2)))#        model.add(Dropout(0.25))                model.add(Conv2D(128, (3, 3), padding='same'))        model.add(Activation('relu'))                model.add(Conv2D(128, (3, 3)))        model.add(Activation('relu'))        model.add(MaxPooling2D(pool_size=(2, 2)))#        model.add(Dropout(0.25))                model.add(Conv2D(256, (3, 3), W_regularizer=l2(0.001)))        model.add(Activation('relu'))        model.add(MaxPooling2D(pool_size=(2, 2)))#        model.add(Dropout(0.25))        #        model.add(Conv2D(256, (3, 3), W_regularizer=l2(0.001)))#        model.add(Activation('relu'))#        model.add(MaxPooling2D(pool_size=(2, 2)))#        model.add(Dropout(0.25))                model.add(Flatten())                model.add(Dense(512, W_regularizer=l2(0.005)))        model.add(Activation('relu'))#        model.add(Dropout(0.5))                model.add(Dense(512, W_regularizer=l2(0.005)))        model.add(Activation('relu'))#        model.add(Dropout(0.5))                model.add(Dense(len(labels.keys())))        model.add(Activation('softmax'))
        rms = optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-08, decay=self.weight_decay)        model.compile(loss='binary_crossentropy',                      optimizer=rms,                      metrics= ['accuracy'])
        callbacks = [                EarlyStopping(monitor='val_loss', patience= 20, verbose=1),                ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=1)]
        if not self.data_augmentation:            history = model.fit(x = X_train, y= Y_train, validation_data=(X_valid, Y_valid),                          batch_size=self.batch_size, verbose=2, nb_epoch=self.nb_epoch, callbacks=callbacks,                          shuffle=True, class_weight=dict_weights)
        else:            X_train_aug, X_test_aug = self.setup_generator(X_train, X_valid, Y_train, Y_valid, batch_size=self.batch_size)                history = model.fit_generator(X_train_aug, steps_per_epoch= 50, nb_epoch=self.nb_epoch,                                 shuffle=True, validation_data = (X_valid, Y_valid), callbacks=callbacks)
        p_valid = model.predict(X_valid, batch_size = self.batch_size, verbose=2)        print("True accuracy is : %s"%np.mean(np.equal(np.argmax(p_valid,1), np.argmax(Y_valid,1))))
        # summarize history for accuracy        plt.figure(figsize = (12,12))        plt.plot(history.history['acc'])        plt.plot(history.history['val_acc'])        plt.title('model accuracy')        plt.ylabel('accuracy')        plt.xlabel('epoch')        plt.legend(['train', 'test'], loc='upper left')        plt.show()                # summarize history for loss        plt.figure(figsize = (12,12))        plt.plot(history.history['loss'])        plt.plot(history.history['val_loss'])        plt.title('model loss')        plt.ylabel('loss')        plt.xlabel('epoch')        plt.legend(['train', 'test'], loc='upper left')        plt.show()
        return model
    def load_data(self, image_paths, label_map):
            nrof_samples = len(image_paths)            print("number images size %s" %str(nrof_samples))            img_list = []            y_list = []
            for i in range(nrof_samples):                try:                    img=self.image_descriptors(image_paths[i])                    img_list.append(self.prewhiten(img))                    targets = np.zeros(len(label_map.keys()))                    targets[label_map[os.path.dirname(image_paths[i]).replace(self.path_client + "\\", '')]] = 1                    y_list.append(targets)                    if i%100 ==0:                       print("image  %i/%i with shape %s"%(i, nrof_samples, img_list[i].shape))
                except Exception as e:                    print("could not open picture %s"%image_paths[i])                    print(e)                    pass
            return np.array(img_list), np.array(y_list)        
    def image_descriptors(self, X):                image= Image.open(X)#        im = imresize(np.array(image), (image_size, image_size))        im = imresize(np.array(image), (self.image_size, self.image_size))        return im
    def prewhiten(self, x):        mean = np.mean(x)        std = np.std(x)        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y    
def get_layer_outputs(model, image):    test_image = image    outputs    = [layer.output for layer in model.layers]          # all layer outputs    comp_graph = [K.function([model.input]+ [K.learning_phase()], [output]) for output in outputs]  # evaluation functions
    # Testing    layer_outputs_list = [op([test_image, 1.]) for op in comp_graph]    layer_outputs = []
    for layer_output in layer_outputs_list:        print(layer_output[0][0].shape, end='\n-------------------\n')        layer_outputs.append(layer_output[0][0])
    return layer_outputs
def plot_layer_outputs(layer_number, model,  image):        layer_outputs = get_layer_outputs(model, image)
    x_max = layer_outputs[layer_number].shape[0]    y_max = layer_outputs[layer_number].shape[1]    n     = layer_outputs[layer_number].shape[2]
    L = []    for i in range(n):        L.append(np.zeros((x_max, y_max)))
    for i in range(n):        for x in range(x_max):            for y in range(y_max):                L[i][x][y] = layer_outputs[layer_number][x][y][i]
    for img in L:        plt.figure()        plt.imshow(img, interpolation='nearest')            if __name__ == "__main__":    t = Train_keras()        image = Image.open(r"U:\Technical\Actuarial Research\0330_Image Recognition\data\frontcrash\batch1 (2).jpg")    im = imresize(np.array(image), (128, 128))[:,:,0:3]         im= np.expand_dims(im, 0)                plot_layer_outputs(1, t.model,  im)    
