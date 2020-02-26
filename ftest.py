#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential, Model,load_model
import math, os, json, sys, re
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
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


from keras.applications import VGG16
import numpy as np
from scipy import ndimage

categories=['carry','clapHands','pickUp','pull','push','sitDown','standUp','throw','walk','waveHands']

def dt(folder1,cat):
    
    dataset=[]
    
    j = 0
    y=[]
    for category in categories:
        for file in os.listdir(folder1+'/'+str(category)):
            #print(file)
            img = load_img(folder1+'/'+str(category)+'/'+file)
            img.thumbnail((224, 224))
            img = img.resize((224,224), Image.ANTIALIAS)
            # Convert to Numpy Array
            x = img_to_array(img)  
            x = x.reshape((224,224,3))
            #print(x.shape)
            #im = Image.open(folder+'/'+str(category)+'/'+file)
            #im=cv2.imread(file)
            #cv2.resize(im, (224,224), interpolation=cv2.INTER_CUBIC)
            #im.resize((224, 224))
            #print('hello1')
            #im.load()
            #print('hello2')
            #data = np.asarray(im,dtype="float32")
            #print(data.shape)
            #print("hello1")
            #data= np.array(im)
            #print(data.shape)
            #x = data.reshape((3,224,224))
            dataset.append(x)
            #print(categories.index(category))
            y.append(categories.index(category))

    dataset=np.array(dataset)
    values = np.array(y)
    #print("All images to array!")
    return dataset,values


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder










model_path1 = '/home/ws2/Documents/finaldata1/seg1/train/head_spinal/models/best_weights2.hdf5'

model1 = load_model(model_path1)
dataset1,values=dt('/home/ws2/Documents/finaldata1/seg1/valid/head_spinal',categories)
model1_pred = model1.predict(dataset1,batch_size=64)
classes = np.argmax(model1_pred,axis=1)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

loss, acc = model1.evaluate(dataset1, onehot_encoded)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
#print(accuracy_score(values,classes))
# print(model_v3.evaluate(X_HPM_test,Y_HPM_test,batch_size=5))

model_path2 = '/home/ws2/Documents/finaldata1/seg1/train/left_hand/models/best_weights2.hdf5'
model2 = load_model(model_path2)
dataset2,values=dt('/home/ws2/Documents/finaldata1/seg1/valid/left_hand',categories)
model2_pred = model2.predict(dataset2,batch_size=64)
classes = np.argmax(model2_pred,axis=1)
loss, acc = model2.evaluate(dataset2, onehot_encoded)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
#print(accuracy_score(values,classes))
# print(model_v3.evaluate(X_HPM_test,Y_HPM_test,batch_size=5))

model_path3 = '/home/ws2/Documents/finaldata1/seg1/train/left_leg/models/best_weights2.hdf5'
model3 = load_model(model_path3)
dataset3,values=dt('/home/ws2/Documents/finaldata1/seg1/valid/left_leg',categories)
model3_pred = model3.predict(dataset3,batch_size=64)
classes = np.argmax(model3_pred,axis=1)
loss, acc = model3.evaluate(dataset3, onehot_encoded)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
#print(accuracy_score(values,classes))
# print(model_v3.evaluate(X_HPM_test,Y_HPM_test,batch_size=5))

model_path4 = '/home/ws2/Documents/finaldata1/seg1/train/right_hand/models/best_weights2.hdf5'
model4 = load_model(model_path4)
dataset4,values=dt('/home/ws2/Documents/finaldata1/seg1/valid/right_hand',categories)
model4_pred = model4.predict(dataset4,batch_size=64)
classes = np.argmax(model4_pred,axis=1)
loss, acc = model4.evaluate(dataset4, onehot_encoded)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
#print(accuracy_score(values,classes))
# print(model_v3.evaluate(X_HPM_test,Y_HPM_test,batch_size=5))
model_path5 = '/home/ws2/Documents/finaldata1/seg1/train/right_leg/models/best_weights2.hdf5'
model5 = load_model(model_path5)
dataset5,values=dt('/home/ws2/Documents/finaldata1/seg1/valid/right_leg',categories)
model5_pred = model5.predict(dataset5,batch_size=64)
classes = np.argmax(model5_pred,axis=1)
loss, acc = model5.evaluate(dataset5, onehot_encoded)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

from sklearn.metrics import accuracy_score

import pandas as pd








# Y_test = label_binarize(Y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])=onehot_encoded

## ROC curve for max late fusion


# In[ ]:




