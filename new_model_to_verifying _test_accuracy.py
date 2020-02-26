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


from keras.applications import VGG16
import numpy as np
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img




class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        
        
        
IMAGE_SIZE = [224, 224]  # we will keep the image size as (64,64). You can increase the size for better results. 

# loading the weights of VGG16 without the top layer. These weights are trained on Imagenet dataset.
vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)  # input_shape = (64,64,3) as required by VGG








def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=False, batch_size=4, class_mode='categorical',
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




#path = "/home/ws2/Documents/cropdata/"
path ="/home/ws2/Documents/kinectdata/seg1/"
#path ="/home/ws2/Documents/kinectdata/seg2/"
#path ="/home/ws2/Documents/kinectdata/seg3/"
#path ="/home/ws2/Documents/kinectdata/seg4/"
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

test_data= val_data
# In[6]:


trn_data.shape
#
# In[7]:

#test_y=np.array(test_y)
# save_array(model_path+'train_data.bc', trn_data)
# save_array(model_path+'valid_data.bc', val_data)

# trn_data = load_array(model_path+'train_data.bc')
# val_data = load_array(model_path+'valid_data.bc')

print(val_data.shape)
print(trn_data.shape)
#######b=np.array(val_batches)
#######print(b.shape)
#####print(val_batches.shape)

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

filenames= val_batches.filenames
nb_samples = len(filenames)
print(nb_samples)
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
    dtype=np.float32)
    dataset=[]
    j = 0
    y=[]
    for category in categories:
        for file in os.listdir(folder1+'/'+str(category)):
            print(file)
            img = load_img(folde1r+'/'+str(category)+'/'+file)
            img.thumbnail((image_width, image_height))
            img = img.resize((224,224), Image.ANTIALIAS)
            # Convert to Numpy Array
            x = img_to_array(img)  
            x = x.reshape((224,224,3))
            print(x.shape)
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
            print(categories.index(category))
            y.append(categories.index(category))

    dataset=np.array(dataset)
    values = np.array(y)
    #print("All images to array!")
    return dataset,values

print(y)

#print(dataset.shape)
#print(y.shape)
# In[ ]:
#y_labels=onehot(y)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
dataset=np.array(dataset)
values = np.array(y)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded.shape)
test_pred=model.predict(dataset)
print(test_pred.shape)
print(test_pred[0])
test_pred1=np.argmax(test_pred, axis=1)
######print(val_classes[0])
print(onehot_encoded[0])
#from sklearn import metrics
from sklearn.metrics import accuracy_score
print(dataset.shape)
print(onehot_encoded.shape)




# In[ ]:


def fit_model(model, batches, val_batches, nb_epoch,saveBestModel):
    model.fit_generator(batches,960,  epochs=nb_epoch, 
                        validation_data=val_batches,nb_val_samples=val_batches.n,callbacks=[saveBestModel,TestCallback((dataset, onehot_encoded))])
    
opt = RMSprop(lr=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])



##best_weights_filepath = model_path+'best_weights1.hdf5'
#earlyStopping=kcallbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
saveBestModel = kcallbacks.ModelCheckpoint(best_weights_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
nb_epoch=50
fit_model(model, batches, val_batches, nb_epoch,saveBestModel)
model.save_weights(model_path+'after_test_weights.hdf5')


# In[ ]:


# # val_batches = get_batches(path+'valid', shuffle=False, batch_size=4)
# batches = get_batches(path+'train', shuffle=False, batch_size=4)


# # In[3]:


# val_data = get_data(path+'valid')

# trn_data = get_data(path+'train')

# test_data= val_data
# # In[6]:


# trn_data.shape
# #
# # In[7]:

# #test_y=np.array(test_y)
# # save_array(model_path+'train_data.bc', trn_data)
# # save_array(model_path+'valid_data.bc', val_data)

# # trn_data = load_array(model_path+'train_data.bc')
# # val_data = load_array(model_path+'valid_data.bc')

# print(val_data.shape)
# print(trn_data.shape)
# #######b=np.array(val_batches)
# #######print(b.shape)
# #####print(val_batches.shape)

# # In[8]:


# def onehot(x): return np.array(OneHotEncoder().fit_transform(x.reshape(-1,1)).todense())
# val_classes = val_batches.classes
# trn_classes = batches.classes
# val_labels = onehot(val_classes)
# trn_labels = onehot(trn_classes)


# # In[9]:


# trn_labels.shape


# # In[10]:


# trn_classes[:4]


# # In[11]:


# trn_labels[:4]


# # In[12]:

# filenames= val_batches.filenames
# nb_samples = len(filenames)
# print(nb_samples)
# model.summary()

