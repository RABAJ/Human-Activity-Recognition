#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from keras.layers import Input, Conv3D, Activation, BatchNormalization, GlobalAveragePooling3D, Dense, Dropout,MaxPooling3D,Add,Convolution3D
from keras.layers.merge import add



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

#base_model = ResNet50(input_shape = IMAGE_SIZE + [3],include_top=False )






# def save_array(fname, arr):
#     c=bcolz.carray(arr, rootdir=fname, mode='w')
#     c.flush()


# def load_array(fname):
#     return bcolz.open(fname)[:]



#path = "/home/ws2/Documents/cropdata/"
#path ="/home/ws2/Documents/finaldata1/seg4/train/head_spinal/"
#path ="/home/ws2/Documents/finaldata1/seg4/train/left_hand/"
#path ="/home/ws2/Documents/finaldata1/seg4/train/left_leg/"
#path ="/home/ws2/Documents/finaldata1/seg4/train/right_hand/"
#path ="/home/ws2/Documents/finaldata1/seg4/train/right_leg/"
#path ="/home/ws2/Documents/traintest_dataparts/seg1/train/head_spinal/"
path ="/home/ws2/Documents/finalcropskdataset1/seg1/train/"
#path ="/home/ws2/Documents/traintest_dataparts/seg1/train/left_hand/"
#path ="/home/ws2/Documents/traintest_dataparts/seg1/train/left_leg/"
#path ="/home/ws2/Documents/traintest_dataparts/seg1/train/right_hand/"
#path ="/home/ws2/Documents/traintest_dataparts/seg1/train/right_leg/"
#path ="/home/ws2/Documents/kinectdata/seg2/"
#path ="/home/ws2/Documents/kinectdata/seg3/"
#path ="/home/ws2/Documents/kinectdata/seg4/"
# path = "data/dogscats/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)
    
#batch_size=4

# from vgg16 import Vgg16
# vgg = Vgg16()
#model = vgg
#model = base_model

# In[2]:


# # In[3]:


# # In[6]:

#model.summary()


# In[13]:


#model.layers.pop()
#for layer in model.layers: layer.trainable=True

# trn_data can be array of images in integer form 
# from keras.models import Sequential
from keras.layers import Dense, Conv3D, BatchNormalization, Activation
from keras.layers import AveragePooling3D, Input, Flatten
from keras.optimizers import Adam,SGD
# print(model.layers[-1].output.shape)
# h = Flatten()(model.layers[-1].output)
# h = BatchNormalization()(h)
# h = Dense(512, activation='relu')(h)
# h = BatchNormalization()(h)
# h = Activation('relu')(h)
#h=Dense(10, activation='softmax')(h)
from keras.models import Model
#model = Model(input=model.input,output=h)





input_tensor = Input((224, 224,50,3))



class Residual(Layer):
    def __init__(self, channels_in,kernel,**kwargs):
        super(Residual, self).__init__(**kwargs)
        self.channels_in = channels_in
        self.kernel = kernel

    def call(self, x):
        # the residual block using Keras functional API
        
        h =             Convolution3D(self.channels_in,
                               self.kernel,
                               strides=(1,1,1),
                               padding="same")(x)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        
        h =             Convolution3D( self.channels_in,
                                self.kernel,
                                strides=(1,1,1),
                                padding="same")(h)
        h = BatchNormalization()(h)
        residual =      Add()([x,h])
        x =             Activation("relu")(residual)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape



model=Sequential()
model.add(Convolution3D(64,kernel_size=(3,3,3),strides=(1,1,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2,2,2)))
#model.add(Residual(64,(3,3,3)))
model.add(Convolution3D(128,kernel_size=(3,3,3),strides=(1,1,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2,2,2)))
#model.add(Residual(128,(3,3,3)))
model.add(Convolution3D(256,kernel_size=(3,3,3),strides=(1,1,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2,2,2)))
#model.add(Residual(256,(3,3,3)))
model.add(Convolution3D(512,kernel_size=(3,3,3),strides=(1,1,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2,2,2)))
#model.add(Residual(512,(3,3,3)))
model.add(Convolution3D(1024,kernel_size=(3,3,3),strides=(1,1,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2,2,2)))
#model.add(Residual(1024,(3,3,3)))
model.add(GlobalAveragePooling3D())
#x=Dropout(0.3)(x)
model.add(BatchNormalization())
model.add(Dense(512))
#x=Dropout(0.3)(x)
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(256))
#x=Dropout(0.3)(x)
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

# model = Model(inputs=input_tensor, outputs=output)

# In[14]:


gen=image.ImageDataGenerator()
folder = path+'valid'
categories=['carry','clapHands','pickUp','pull','push','sitDown','standUp','throw','walk','waveHands']

training_data= []

#onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train_files = []
y_train = []
i=0


def dt(folder1,cat):
    
    dataset=[]
    j = 0
    ii=0
    y1=[]
    for category in categories:
        

        for i in range(1,21): 
            if(os.path.exists(folder1+'/'+str(category)+'/'+'video'+str(i))):
                dataset1=[]
                d=[]
                for file in os.listdir(folder1+'/'+str(category)+'/'+'video'+str(i)):
        #print(file)
                    ii=ii+1
                    img = load_img(folder1+'/'+str(category)+'/'+'video'+str(i)+'/'+file)
                    img.thumbnail((224,224))
                    img = img.resize((224,224), Image.ANTIALIAS)
                # Convert to Numpy Array
                    x = img_to_array(img)  
                    x = x.reshape((224,224,3))

                    dataset1.append(x)
                #print(categories.index(category))
                if(len(dataset1)!=0):
                    y1.append(categories.index(category))
                    if(len(dataset1)<=50):
                        f=50//len(dataset1)
                        g=50-f*len(dataset1)
                        p=len(dataset1)-g
                        for i in range(0,len(dataset1)):
                            if(len(dataset1)!=50):
                                if(i<p):

                                    for j in range(0,f):
                                        d.append(dataset1[i])
                                else:
                                    for j in range(0,f+1):
                                        d.append(dataset1[i])

                            else:

                                d.append(dataset1[i])
                    hh=0
                    if(len(dataset1)>50):
                        h=len(dataset1)-50
                        print(len(dataset1))
                        print(h)

                        l=len(dataset1)//h
                        if(l<=2):
                            l=3
                        print(l)
                        for i in range(0,len(dataset1)):
                            if(hh<h):
                                if(i%(l-1)==0):

                                    hh=hh+1
                                    continue
                            d.append(dataset1[i])
                        if(hh<h):
                            hs=0
                            d1=[]
                            s=len(d)-50
                            l=len(d)//s
                            if(l<=2):
                                l=3
                            print(l)
                            for i in range(0,len(d)):
                                if(hs<s):
                                    if(i%(l-1)==0):

                                        hs=hs+1
                                        
                                        continue
                                d1.append(d[i])
                            d=d1
                    d=np.asarray(d,dtype=np.float32)
                    d=d.reshape((224,224,-1,3))
                    print(d.shape)
                    dataset.append(d)
            #print('yes',ii)
    dataset=np.asarray(dataset,dtype=np.float32)
    print('ye')
    values = np.array(y1)
    #print("All images to array!")
    return dataset,values

#print(y)
dataset,values=dt('/home/ws2/Documents/finalcropskdataset1/seg1/train',categories)
print(dataset.shape)
#print(values.shape)
dataset1,values1=dt('/home/ws2/Documents/finalcropskdataset1/seg1/valid',categories)

# In[ ]:
#y_labels=onehot(y)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
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
best_weights_filepath = model_path+'best_weights5.hdf5'
#earlyStopping=kcallbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
saveBestModel = kcallbacks.ModelCheckpoint(best_weights_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

#callbacks=[saveBestModel,p]
aug = ImageDataGenerator(rotation_range=0, zoom_range=0,
                         width_shift_range=0, height_shift_range=0, shear_range=0,
                         horizontal_flip=False)
print(len(dataset1))
def fit_model(model, dataset1,onehot_encoded1,dataset,onehot_encoded):
    history = model.fit(dataset,onehot_encoded,
                            validation_data=(dataset1,onehot_encoded1), steps_per_epoch=len(dataset) //32,
                            epochs=200,callbacks=[saveBestModel,TestCallback((dataset, onehot_encoded))])
    print(history.history.keys())
    # summarize history for accuracy
    acc=(history.history['acc'])
    np.save('/home/ws2/Documents/newmodelsarray1/acc_seg1.npy', acc)
    val_acc=(history.history['val_acc'])
    np.save('/home/ws2/Documents/newmodelsarray1/val_acc_seg1.npy', val_acc)
    loss=(history.history['loss'])
    np.save('/home/ws2/Documents/newmodelsarray1/loss_seg1.npy', loss)
    val_loss=(history.history['val_loss'])
    np.save('/home/ws2/Documents/newmodelsarray1/val_loss_seg1.npy', val_loss)

    
opt = SGD(lr=0.0005,momentum=0.9,decay=1.0e-6)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])




fit_model(model, dataset1,onehot_encoded1,dataset,onehot_encoded)
#model.save_weights(model_path+'after_test_weights.hdf5')


# In[ ]:



