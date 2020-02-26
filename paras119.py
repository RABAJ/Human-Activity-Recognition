#!/usr/bin/env python
# coding: utf-8

# In[14]:


# import re
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
subsets=['train','valid']
waste_types=['A001','A002','A003','A004','A005','A006','A007','A008','A009','A010','A011','A012','A013','A014','A015','A016','A017','A018','A019','A020','A021','A022','A023','A024','A025','A026','A027','A028','A029','A030','A031','A032','A033','A034','A035','A036','A037','A038','A039','A040','A041','A042','A043','A044','A045','A046','A047','A048','A049','A050','A051','A052','A053','A054','A055','A056','A057','A058','A059','A060']

for subset in subsets:
    for waste_type in waste_types:
        folder=os.path.join('/home/ws2/Documents','fidntu_lefthand',subset,waste_type)
        if not os.path.exists(folder):
            os.makedirs(folder)

            
            
for file in os.listdir('/home/ws2/Documents/nturgb_lefthand'):
    if('C001' in file):
        title=file[16:-4]
        title1=file[:-4]
        f=os.path.join('/home/ws2/Documents/nturgb_lefthand',file)
        shutil.copy(f,os.path.join('/home/ws2/Documents/fidntu_lefthand','valid',title))
        img = Image.open(f)

        img2 = img.transpose(Image.FLIP_LEFT_RIGHT)
        img2.save('/home/ws2/Documents/fidntu_lefthand/valid/'+title+'/'+title1+str('_1')+'.png')
        
    elif('C002' in file or 'C003' in file):
        title=file[16:-4]
        title1=file[:-4]
        f=os.path.join('/home/ws2/Documents/nturgb_lefthand',file)
        shutil.copy(f,os.path.join('/home/ws2/Documents/fidntu_lefthand','train',title))
        
#         img2 = np.fliplr(f) 
        

        img = Image.open(f)

        img2 = img.transpose(Image.FLIP_LEFT_RIGHT)
        img2.save('/home/ws2/Documents/fidntu_lefthand/train/'+title+'/'+title1+str('_1')+'.png')


# In[ ]:





# In[ ]:




