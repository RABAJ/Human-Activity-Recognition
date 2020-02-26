#!/usr/bin/env python
# coding: utf-8

# In[36]:




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


# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)








model_path1 = '/home/ws2/Documents/kinectdata/seg1/models/best_weights1.hdf5'
model1 = load_model(model_path1)
dataset1,values=dt('/home/ws2/Documents/kinectdata/seg1/valid',categories)
model1_pred = model1.predict(dataset1,batch_size=4)
classes = np.argmax(model1_pred,axis=1)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

loss, acc = model1.evaluate(dataset1, onehot_encoded, verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
#print(accuracy_score(values,classes))
# print(model_v3.evaluate(X_HPM_test,Y_HPM_test,batch_size=5))

model_path2 = '/home/ws2/Documents/kinectdata/seg2/models/best_weights1.hdf5'
model2 = load_model(model_path2)
dataset2,values=dt('/home/ws2/Documents/kinectdata/seg2/valid',categories)
model2_pred = model1.predict(dataset2,batch_size=4)
classes = np.argmax(model2_pred,axis=1)
loss, acc = model2.evaluate(dataset2, onehot_encoded, verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
#print(accuracy_score(values,classes))
# print(model_v3.evaluate(X_HPM_test,Y_HPM_test,batch_size=5))

model_path3 = '/home/ws2/Documents/kinectdata/seg3/models/best_weights1.hdf5'
model3 = load_model(model_path3)
dataset3,values=dt('/home/ws2/Documents/kinectdata/seg3/valid',categories)
model3_pred = model3.predict(dataset3,batch_size=4)
classes = np.argmax(model3_pred,axis=1)
loss, acc = model3.evaluate(dataset3, onehot_encoded, verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
#print(accuracy_score(values,classes))
# print(model_v3.evaluate(X_HPM_test,Y_HPM_test,batch_size=5))

model_path4 = '/home/ws2/Documents/kinectdata/seg4/models/best_weights1.hdf5'
model4 = load_model(model_path4)
dataset4,values=dt('/home/ws2/Documents/kinectdata/seg4/valid',categories)
model4_pred = model4.predict(dataset4,batch_size=4)
classes = np.argmax(model4_pred,axis=1)
loss, acc = model4.evaluate(dataset4, onehot_encoded, verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
#print(accuracy_score(values,classes))
# print(model_v3.evaluate(X_HPM_test,Y_HPM_test,batch_size=5))


from sklearn.metrics import accuracy_score

import pandas as pd








# Y_test = label_binarize(Y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])=onehot_encoded

## ROC curve for max late fusion


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
####model.load_weights("/home/ws2/Documents/kinectdata/seg1/models/best_weights1.hdf5")
#####model = load_model('/home/ws2/Documents/kinectdata/seg1/models/best_weights1.hdf5')
#####filenames= val_batches.filenames
#####nb_samples = len(filenames)

# In[ ]:


# Original Dimensions
image_width = 224
image_height = 224
#ratio = 4




#dataset = np.ndarray(shape=(len(train_files), channels, image_height, image_width),
#  



print(accuracy_score(values,test_pred1))


# In[31]:


print('weighted average')

np.random.seed(123)

df = pd.DataFrame(columns=('w1', 'w2', 'w3','w4', 'accuracy'))
prediction= (model1_pred+model2_pred+model3_pred+model4_pred)/4.0
classes_pro = prediction_weight_avg.argmax(axis=-1)
print(accuracy_score(values,classes_pro))
print(prediction[0])
i = 0
for w1 in range(1,5):
    for w2 in range(1,5):
        for w3 in range(1,5):
            for w4 in range(1,5):
                w=(w1+w2+w3+w4)/4.0
                prediction_weight_avg = (w1*model1_pred+w2*model2_pred+w3*model3_pred+w4*model4_pred)/w
                print(prediction_weight_avg[0])
                classes_pro = np.argmax(prediction_weight_avg,axis=-1)
                ac1=accuracy_score(values,classes_pro)
                #print(ac1)
                df.loc[i] = [w1, w2, w3,w4, ac1]
                i += 1
print(ac1)
#df.sort(columns=['accuracy'], ascending=False)

df.sort_values(by='accuracy',ascending=False)
https://sebastianraschka.com/Articles/2014_ensemble_classifier.html


# In[30]:


from sklearn.metrics import roc_curve, auc

# print("fused max")
# prediction_max = np.maximum(model1_pred,model2_pred,model3_pred,model4_pred)
# classes_max = prediction_max.argmax(axis=-1)
# print(accuracy_score(values,classes_max))
print("fused avg")
prediction_avg = model1_pred+model2_pred+model3_pred+model4_pred/4.0
classes_avg =np.argmax(prediction_avg,axis=-1) 
print(accuracy_score(values,classes_avg))
print("fused product")
prediction_pro = (model1_pred*model2_pred*model3_pred*model4_pred)/4.0
classes_pro = np.argmax(prediction_pro,axis=-1) 
print(accuracy_score(values,classes_pro))

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)






n_classes = 10
# fpr_max = dict()
# tpr_max = dict()
# roc_auc_max = dict()

fpr_avg = dict()
tpr_avg = dict()
roc_auc_avg = dict()

fpr_and = dict()
tpr_and = dict()
roc_auc_and = dict()

fpr_model1 = dict()
tpr_model1 = dict()
roc_auc_model1 = dict()

fpr_model2 = dict()
tpr_model2 = dict()
roc_auc_model2 = dict()
fpr_model3 = dict()
tpr_model3 = dict()
roc_auc_model3 = dict()
fpr_model4 = dict()
tpr_model4 = dict()
roc_auc_model4 = dict()

# for i in range(n_classes):
#     fpr_max[i], tpr_max[i], _ = roc_curve(onehot_encoded[:, i], prediction_max[:, i])
#     roc_auc_max[i] = auc(fpr_max[i], tpr_max[i])

# # Compute micro-average ROC curve and ROC area for max late fusion
# fpr_max["micro"], tpr_max["micro"], _ = roc_curve(onehot_encoded.ravel(), prediction_max.ravel())
# roc_auc_max["micro"] = auc(fpr_max["micro"], tpr_max["micro"])

#########################################################
for i in range(n_classes):
    fpr_model1[i], tpr_model1[i], _ = roc_curve(onehot_encoded[:, i], model1_pred[:, i])
    roc_auc_model1[i] = auc(fpr_model1[i], tpr_model1[i])

# Compute micro-average ROC curve and ROC area for model1
fpr_model1["micro"], tpr_model1["micro"], _ = roc_curve(onehot_encoded.ravel(), model1_pred.ravel())
roc_auc_model1["micro"] = auc(fpr_model1["micro"], tpr_model1["micro"])

for i in range(n_classes):
    fpr_model2[i], tpr_model2[i], _ = roc_curve(onehot_encoded[:, i], model2_pred[:, i])
    roc_auc_model2[i] = auc(fpr_model2[i], tprmodel2[i])

# Compute micro-average ROC curve and ROC area for model2
fpr_model2["micro"], tpr_model2["micro"], _ = roc_curve(onehot_encoded.ravel(), model2_pred.ravel())
roc_auc_model2["micro"] = auc(fpr_model2["micro"], tpr_model2["micro"])

for i in range(n_classes):
    fpr_model3[i], tpr_model3[i], _ = roc_curve(onehot_encoded[:, i], model3_pred[:, i])
    roc_auc_model3[i] = auc(fpr_model3[i], tprmodel3[i])

# Compute micro-average ROC curve and ROC area for model3
fpr_model3["micro"], tpr_model3["micro"], _ = roc_curve(onehot_encoded.ravel(), model3_pred.ravel())
roc_auc_model3["micro"] = auc(fpr_model3["micro"], tpr_model3["micro"])


for i in range(n_classes):
    fpr_model4[i], tpr_model4[i], _ = roc_curve(onehot_encoded[:, i], model4_pred[:, i])
    roc_auc_model4[i] = auc(fpr_model4[i], tprmodel4[i])

# Compute micro-average ROC curve and ROC area for model4
fpr_model4["micro"], tpr_model4["micro"], _ = roc_curve(onehot_encoded.ravel(), model4_pred.ravel())
roc_auc_model4["micro"] = auc(fpr_model4["micro"], tpr_model4["micro"])






for i in range(n_classes):
    fpr_avg[i], tpr_avg[i], _ = roc_curve(onehot_encoded[:, i], prediction_avg[:, i])
    roc_auc_avg[i] = auc(fpr_avg[i], tpr_avg[i])

# Compute micro-average ROC curve and ROC area for avg late fusion
fpr_avg["micro"], tpr_avg["micro"], _ = roc_curve(onehot_encoded.ravel(), prediction_avg.ravel())
roc_auc_avg["micro"] = auc(fpr_avg["micro"], tpr_avg["micro"])


for i in range(n_classes):
    fpr_and[i], tpr_and[i], _ = roc_curve(onehot_encoded[:, i], prediction_pro[:, i])
    roc_auc_and[i] = auc(fpr_and[i], tpr_and[i])

# Compute micro-average ROC curve and ROC area for product late fusion
fpr_and["micro"], tpr_and["micro"], _ = roc_curve(onehot_encoded.ravel(), prediction_pro.ravel())
roc_auc_and["micro"] = auc(fpr_and["micro"], tpr_and["micro"])

 #Plot ROC curves for the multiclass problem

# Compute macro-average ROC curve and ROC area

# # #### First aggregate all false positive rates
# #
# #
# #
# all_fpr_max = np.unique(np.concatenate([fpr_max[i] for i in range(n_classes)]))
# all_fpr_avg = np.unique(np.concatenate([fpr_avg[i] for i in range(n_classes)]))
# all_fpr_and = np.unique(np.concatenate([fpr_and[i] for i in range(n_classes)]))
# # Then interpolate all ROC curves at this points
# mean_tpr_max = np.zeros_like(all_fpr_max)
# mean_tpr_avg = np.zeros_like(all_fpr_avg)
# mean_tpr_and = np.zeros_like(all_fpr_and)
# for i in range(n_classes):
#     mean_tpr_max += interp(all_fpr_max, fpr_max[i], tpr_max[i])
# for i in range(n_classes):
#     mean_tpr_avg += interp(all_fpr_avg, fpr_avg[i], tpr_avg[i])
# for i in range(n_classes):
# 	mean_tpr_and += interp(all_fpr_and, fpr_and[i], tpr_and[i])
#
# # Finally average it and compute AUC
# mean_tpr_max /= n_classes
# mean_tpr_avg /= n_classes
# mean_tpr_and /= n_classes
#
# fpr_max["macro"] = all_fpr_max
# tpr_max["macro"] = mean_tpr_max
# roc_auc_max["macro"] = auc(fpr_max["macro"], tpr_max["macro"])
#
# fpr_avg["macro"] = all_fpr_avg
# tpr_avg["macro"] = mean_tpr_avg
# roc_auc_avg["macro"] = auc(fpr_avg["macro"], tpr_avg["macro"])
#
# fpr_and["macro"] = all_fpr_and
# tpr_and["macro"] = mean_tpr_and
# roc_auc_and["macro"] = auc(fpr_and["macro"], tpr_and["macro"])

# Plot all ROC curves

plt.figure(dpi=100)
# plt.plot(fpr_max["micro"], tpr_max["micro"],
#          label='max late fusion(area = {0:0.2f})'
#                ''.format(roc_auc_max["micro"]),
#          color='deeppink', linestyle=':', linewidth=2)
# #
# plt.plot(fpr_avg["micro"], tpr_avg["micro"],
#          label='avg late fusion(area = {0:0.2f})'
#                ''.format(roc_auc_avg["micro"]),
#          color='navy', linestyle='-', linewidth=2)

print(len(fpr_and["micro"]))
print(len(tpr_and["micro"]))
print(len(fpr_hpm["micro"]))
print(len(tpr_hpm["micro"]))

# import scipy.io as sio
# sio.savemat('/home/chavvi/Documents/chhavi/roc_and_view2.mat',dict([('FPR_and',fpr_and["micro"]),('TPR_and',tpr_and["micro"])]),appendmat=True,format='5',long_field_names=False,do_compression=False,oned_as='row')
# sio.savemat('/home/chavvi/Documents/chhavi/roc_hpm_view2.mat',dict([('FPR_hpm',fpr_hpm["micro"]),('TPR_hpm',tpr_hpm["micro"])]),appendmat=True,format='5',long_field_names=False,do_compression=False,oned_as='row')
# sio.savemat('/home/chavvi/Documents/chhavi/roc_di_view2.mat',dict([('FPR_di',fpr_di["micro"]),('TPR_di',tpr_di["micro"])]),appendmat=True,format='5',long_field_names=False,do_compression=False,oned_as='row')
plt.plot(fpr_and["micro"], tpr_and["micro"],
         label='Late fusion (DI_InceptionV3 + HPM_LSTM)(AUC = {0:0.2f})'
               ''.format(roc_auc_di["micro"]),
         color='green', linestyle='solid', linewidth=1,marker='o',markersize=2)
plt.plot(fpr_hpm["micro"], tpr_hpm["micro"],
         label=' HPM_LSTM (AUC = {0:0.2f})'
               ''.format(roc_auc_hpm["micro"]),
         color='red', linestyle='--', linewidth=1, marker='o',markersize=2)
plt.plot(fpr_di["micro"], tpr_di["micro"],
         label=' DI_InceptionV3 (AUC = {0:0.2f})'
               ''.format(roc_auc_and["micro"]),
         color='navy', linestyle='-.', linewidth=1,marker='o',markersize=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




