#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division,print_function
import math, os, json, sys, re
# import cPickle as pickle
from glob import glob


# from fast.ai.imports import *
import PIL
from PIL import Image
from numpy.random import random, permutation, randn, normal, uniform, choice
from numpy import newaxis
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
# from scipy.ndimage import imread
from sklearn.metrics import confusion_matrix
# import bcolz
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE

from IPython.lib.display import FileLink

# import theano
# from theano import shared, tensor as T
# from theano.tensor.nnet import conv2d, nnet
# from theano.tensor.signal import pool

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model

from keras.layers.core import Flatten, Dense, Dropout, Lambda

from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.optimizers import SGD, RMSprop




from keras.applications import VGG16
import numpy as np
#from keras.preprocessing import image

IMAGE_SIZE = [224, 224]  # we will keep the image size as (64,64). You can increase the size for better results. 

# loading the weights of VGG16 without the top layer. These weights are trained on Imagenet dataset.
vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)  # input_shape = (64,64,3) as required by VGG








def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4, class_mode='categorical',
                target_size=(224,224)):
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


def onehot(x):
    return to_categorical(x)



def get_data(path, target_size=(224,224)):
    batches = get_batches(path, shuffle=False, batch_size=4, class_mode=None, target_size=target_size)
    return np.concatenate([batches.next() for i in range(batches.samples)])


def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


def get_classes(path):
    batches = get_batches(path+'train', shuffle=False, batch_size=4)
    val_batches = get_batches(path+'valid', shuffle=False, batch_size=4)
    test_batches = get_batches(path+'test', shuffle=False, batch_size=4)
    return (val_batches.classes, batches.classes, onehot(val_batches.classes), onehot(batches.classes),
        val_batches.filenames, batches.filenames, test_batches.filenames)




path = "/home/ws2/Documents/cropdata/"
# path = "data/dogscats/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)
    
batch_size=4

# from vgg16 import Vgg16
# vgg = Vgg16()
model = vgg


# In[2]:


# Use batch size of 1 since we're just doing preprocessing on the CPU
val_batches = get_batches(path+'valid', shuffle=False, batch_size=4)
batches = get_batches(path+'train', shuffle=False, batch_size=4)


# In[3]:


val_data = get_data(path+'valid')

trn_data = get_data(path+'train')


# In[6]:


trn_data.shape


# In[7]:


# save_array(model_path+'train_data.bc', trn_data)
# save_array(model_path+'valid_data.bc', val_data)

# trn_data = load_array(model_path+'train_data.bc')
# val_data = load_array(model_path+'valid_data.bc')

val_data.shape


# In[8]:


def onehot(x): return np.array(OneHotEncoder().fit_transform(x.reshape(-1,1)).todense())
val_classes = val_batches.classes
trn_classes = batches.classes
val_labels = onehot(val_classes)
trn_labels = onehot(trn_classes)


# In[9]:


trn_labels.shape


# In[10]:


trn_classes[:4]


# In[11]:


trn_labels[:4]


# In[12]:


model.summary()


# In[13]:


model.layers.pop()
for layer in model.layers: layer.trainable=False

# trn_data can be array of images in integer form 
# from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
print(model.layers[-1].output.shape)
h = BatchNormalization()(model.layers[-1].output)
h = Activation('relu')(h)
h = AveragePooling2D(pool_size=8)(h)
h = Flatten()(h)
h = Dense(4096, activation='relu')(h)
h = Dropout(0.5)(h)
h = Dense(1024, activation='relu')(h)
h = Dropout(0.5)(h)
h=Dense(10, activation='softmax')(h)
from keras.models import Model
model = Model(input=model.input,output=h)


# In[14]:


gen=image.ImageDataGenerator()


# In[ ]:


def fit_model(model, batches, val_batches, nb_epoch=8):
    model.fit_generator(batches, 640, epochs=nb_epoch, 
                        validation_data=val_batches, nb_val_samples=80)
    
opt = RMSprop(lr=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


fit_model(model, batches, val_batches, nb_epoch=50)


# In[ ]:





# In[ ]:




