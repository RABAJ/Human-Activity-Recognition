#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division,print_function
import math, os, json, sys, re
# import cPickle as pickle
from glob import glob


# from fast.ai.imports import *
import PIL
from PIL import Image
import numpy as np
from numpy.random import random, permutation, randn, normal, uniform, choice
from numpy import newaxis
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
# from scipy.ndimage import imread
#from sklearn.metrics import confusion_matrix
# import bcolz
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE

#from IPython.lib.display import FileLink

# import theano
# from theano import shared, tensor as T
# from theano.tensor.nnet import conv2d, nnet
# from theano.tensor.signal import pool

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model,load_model



from keras.layers.core import Flatten, Dense, Dropout, Lambda

from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.optimizers import SGD, RMSprop

import keras.callbacks as kcallbacks


from keras.applications import VGG16,ResNet50
import numpy as np
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input



class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        
        
        
IMAGE_SIZE = [224, 224]  # we will keep the image size as (64,64). You can increase the size for better results. 

# loading the weights of VGG16 without the top layer. These weights are trained on Imagenet dataset.
#vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)  # input_shape = (64,64,3) as required by VGG

#base_model = InceptionResNetV2(input_shape = IMAGE_SIZE + [3],include_top=False )

base_model = ResNet50(input_shape = IMAGE_SIZE + [3],include_top=False )






# def save_array(fname, arr):
#     c=bcolz.carray(arr, rootdir=fname, mode='w')
#     c.flush()


# def load_array(fname):
#     return bcolz.open(fname)[:]



#path = "/home/ws2/Documents/cropdata/"
path ="/home/ws2/Documents/DATA_PARTS/seg1/"
#path ="/home/ws2/Documents/DATA_PARTS/seg2/"
#path ="/home/ws2/Documents/DATA_PARTS/seg3/"
#path ="/home/ws2/Documents/DATA_PARTS/seg4/"
# path = "data/dogscats/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)
    
#batch_size=4

# from vgg16 import Vgg16
# vgg = Vgg16()
#model = vgg
model = base_model








model.layers.pop()
for layer in model.layers: layer.trainable=False

# trn_data can be array of images in integer form 
# from keras.models import Sequential
x1 = Flatten()(model.layers[-1].output)
x1 = BatchNormalization()(x1)
x1 = Dense(512, activation='relu')(x1)
x1 = BatchNormalization()(x1)
x1 = Activation('relu')(x1)
x2 = Flatten()(model.layers[-1].output)
x2 = BatchNormalization()(x2)
x2 = Dense(512, activation='relu')(x2)
x2 = BatchNormalization()(x2)
x2 = Activation('relu')(x2)
x3 = Flatten()(model.layers[-1].output)
x3 = BatchNormalization()(x3)
x3 = Dense(512, activation='relu')(x3)
x3 = BatchNormalization()(x3)
x3 = Activation('relu')(x3)
x4 = Flatten()(model.layers[-1].output)
x4 = BatchNormalization()(x4)
x4 = Dense(512, activation='relu')(x4)
x4 = BatchNormalization()(x4)
x4 = Activation('relu')(x4)
x5 = Flatten()(model.layers[-1].output)
x5 = BatchNormalization()(x5)
x5 = Dense(512, activation='relu')(x5)
x5 = BatchNormalization()(x5)
x5 = Activation('relu')(x5)


from keras.layers.merge import concatenate

# merge input models
merge = concatenate([x1, x2,x3,x4,x5])
# interpretation model
hidden1 = Dense(100, activation='relu')(merge)
hidden2 = Dense(100, activation='relu')(hidden1)
output = Dense(10, activation='softmax')(hidden2)
from keras.models import Model
model = Model(inputs=[model.input,model.input,model.input,model.input,model.input,model.input],outputs=output)

# model = Model(inputs=[visible1, visible2], outputs=output)
# summarize layers
print(model.summary())
# plot graph
# plot_model(model, to_file='multiple_inputs.png')

gen=image.ImageDataGenerator()
folder = path+'valid'
categories=['carry','clapHands','pickUp','pull','push','sitDown','standUp','throw','walk','waveHands']

training_data= []

onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train_files = []
y_train = []
i=0


def dt(folder1,cat):
    
    dataset=[]
    j = 0
    y1=[]
    for category in categories:
        for file in os.listdir(folder1+'/'+str(category)):
            #print(file)
            img = load_img(folder1+'/'+str(category)+'/'+file)
            img.thumbnail((224,224))
            img = img.resize((224,224), Image.ANTIALIAS)
            # Convert to Numpy Array
            x = img_to_array(img)  
            x = x.reshape((224,224,3))
            
            dataset.append(x)
            #print(categories.index(category))
            y1.append(categories.index(category))

    dataset=np.array(dataset)
    values = np.array(y1)
    #print("All images to array!")
    return dataset,values

#print(y)
validdataset1,validvalues1=dt('/home/ws2/Documents/FINAL_DATA/seg1/valid/head_spinal',categories)
validdataset2,validvalues2=dt('/home/ws2/Documents/FINAL_DATA/seg1/valid/left_hand',categories)
validdataset3,validvalues3=dt('/home/ws2/Documents/FINAL_DATA/seg1/valid/left_leg',categories)
validdataset4,validvalues4=dt('/home/ws2/Documents/FINAL_DATA/seg1/valid/right_hand',categories)
validdataset5,validvalues5=dt('/home/ws2/Documents/FINAL_DATA/seg1/valid/right_leg',categories)
#print(dataset.shape)
#print(values.shape)
dataset1,values1=dt('/home/ws2/Documents/FINAL_DATA/seg1/train/head_spinal',categories)
dataset2,values1=dt('/home/ws2/Documents/FINAL_DATA/seg1/train/left_hand',categories)
dataset3,values1=dt('/home/ws2/Documents/FINAL_DATA/seg1/train/left_leg',categories)
dataset4,values1=dt('/home/ws2/Documents/FINAL_DATA/seg1/train/right_hand',categories)
dataset5,values1=dt('/home/ws2/Documents/FINAL_DATA/seg1/train/right_leg',categories)

# In[ ]:
#y_labels=onehot(y)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(validvalues1)
#print(integer_encoded)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

integer_encoded1 = label_encoder.fit_transform(values1)
#print(integer_encoded)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded1 = integer_encoded1.reshape(len(integer_encoded1), 1)
onehot_encoded1 = onehot_encoder.fit_transform(integer_encoded1)





# In[ ]:

#nb_epoch=50
#p=TestCallback((dataset, onehot_encoded))
best_weights_filepath = model_path+'best_weights2.hdf5'
#earlyStopping=kcallbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
# saveBestModel = kcallbacks.ModelCheckpoint(best_weights_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

#callbacks=[saveBestModel,p]
aug = ImageDataGenerator(rotation_range=0, zoom_range=0,
                         width_shift_range=0, height_shift_range=0, shear_range=0,
                         horizontal_flip=False)
# def fit_model(model, dataset1,onehot_encoded1,dataset,onehot_encoded):
#     H = model.fit_generator(aug.flow(dataset1,onehot_encoded1, batch_size=4),
#                             validation_data=(dataset,onehot_encoded), steps_per_epoch=len(dataset1) //4,
#                             epochs=200,callbacks=[TestCallback((dataset, onehot_encoded))])
    

#model.save_weights(model_path+'after_test_weights.hdf5')


def generator_two_img(X1, X2, X3,X4,X5, y, batch_size):
    genX1 = aug.flow(X1, y,  batch_size=4, seed=1)
    genX2 = aug.flow(X2, y, batch_size=4, seed=1)
    genX3 = aug.flow(X3, y, batch_size=4, seed=1)
    genX4 = aug.flow(X4, y, batch_size=4, seed=1)
    genX5 = aug.flow(X5, y, batch_size=4, seed=1)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()
        X4i = genX4.next()
        X5i = genX5.next()
        yield [X1i[0], X2i[0], X3i[0],X4i[0],X5i[0]], X1i[1]
 
 def generator_two_img1(X1, X2, X3,X4,X5, y, batch_size):
    genX1 = aug.flow(X1, y,  batch_size=4, seed=1)
    genX2 = aug.flow(X2, y, batch_size=4, seed=1)
    genX3 = aug.flow(X3, y, batch_size=4, seed=1)
    genX4 = aug.flow(X4, y, batch_size=4, seed=1)
    genX5 = aug.flow(X5, y, batch_size=4, seed=1)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()
        X4i = genX4.next()
        X5i = genX5.next()
        yield [X1i[0], X2i[0], X3i[0],X4i[0],X5i[0]], X1i[1]


def fit_model(model, dataset1,dataset2,dataset3,dataset4,dataset5,onehot_encoded1,validdataset1,validdataset2,validdataset3,validdataset4,validdataset5,onehot_encoded):
    hist = model.fit_generator(generator_two_img(dataset1, dataset2,dataset3,dataset4,dataset5, 
                                                 onehot_encoded1, 4),
                               steps_per_epoch=len(x_train1) // 4, 
                               epochs= 200,
                               callbacks = [],
                               validation_data=generator_two_img1(validdataset1,validdataset2,validdataset3,validdataset4,validdataset5,onehot_encoded,4),
                               validation_steps=dataset.shape[0] // 4, 
                               verbose=1)


opt = Adam(lr=0.0001,decay=1e-6)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])




fit_model(model, dataset1,dataset2,dataset3,dataset4,dataset5,onehot_encoded1,dataset,onehot_encoded)

