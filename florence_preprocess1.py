#!/usr/bin/env python
# coding: utf-8

# In[6]:


# import re
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
# subsets=['train','valid']
waste_types=['a1','a2','a3','a4','a5','a6','a7','a8','a9']


for waste_type in waste_types:
    folder=os.path.join('/home/ws2/Documents','fidflore_righthand_new',waste_type)
    if not os.path.exists(folder):
        os.makedirs(folder)


            
            
for waste in waste_types:
    i=0
    for file in os.listdir('/home/ws2/Documents/florence/flore_righthand1'):
    
        if(waste in file):
            if(file == 'a6s186p9.png' ):
                continue
            title=file[0:2]
            title1=file[:-4]
            f=os.path.join('/home/ws2/Documents/florence/flore_righthand1',file)
            i=i+1
            shutil.copy(f,os.path.join('/home/ws2/Documents/fidflore_righthand_new',title,title1+'_'+str(i)+'.png'))
            
            img = Image.open(f)
            img2 = img.transpose(Image.FLIP_LEFT_RIGHT)
            img2.save('/home/ws2/Documents/fidflore_righthand_new/'+title+'/'+title1+'_'+str(i)+'.png')
            i=i+1
            img3=img.rotate(15)
            img3.save('/home/ws2/Documents/fidflore_righthand_new/'+title+'/'+title1+'_'+str(i)+'.png')
        
    


# In[7]:


from pathlib import Path
from glob2 import glob
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import os
import shutil
import re
import seaborn as sns
import random
subsets=['train','valid']
def split_indices(folder,seed1,seed2):
    n=len(os.listdir(folder))
    full_set=list(range(1,n+1))
    random.seed(seed1)
    train=random.sample(list(range(1,n+1)),int(.8*n))
    remain=list(set(full_set)-set(train))
    random.seed(seed2)
    
    valid = list(set(remain)-set(train))
    
    return(train,valid)

## gets file names for a particular type of trash, given indices
    ## input: waste category and indices
    ## output: file names 
def get_names(s_f,indices):
    file_names=[]
    for file in os.listdir(s_f):
        for i in indices:
#             print(i)
#             print(file)
            if(file.endswith(str(i)+'.png')):
                print(file)
                file_names.append(file)
        
#     print(file_names)
    return(file_names)    

## moves group of source files to another folder
    ## input: list of source files and destination folder
    ## no output
def move_files(source_files,destination_folder):
    for file in source_files:
        shutil.copy(file,destination_folder)
        
for subset in subsets:
    for waste_type in waste_types:
        folder=os.path.join('/home/ws2/Documents','ttfidflore_righthand_new',subset,waste_type)
        if not os.path.exists(folder):
            os.makedirs(folder )       

            
## move files to destination folders for each waste type
for waste_type in waste_types:
    source_folder = os.path.join('/home/ws2/Documents/fidflore_righthand_new',waste_type)
    train_ind, valid_ind= split_indices(source_folder,1,1)
    
    ## move source files to train
    train_names = get_names(source_folder,train_ind)
    train_source_files = [os.path.join(source_folder,name) for name in train_names]
    train_dest = "/home/ws2/Documents/ttfidflore_righthand_new/train/"+waste_type
    move_files(train_source_files,train_dest)
    
    ## move source files to valid
    valid_names = get_names(source_folder,valid_ind)
    valid_source_files = [os.path.join(source_folder,name) for name in valid_names]
    valid_dest = "/home/ws2/Documents/ttfidflore_righthand_new/valid/"+waste_type
    move_files(valid_source_files,valid_dest)
    
    



            
            
        


# In[ ]:




