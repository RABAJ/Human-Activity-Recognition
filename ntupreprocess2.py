#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from mpl_toolkits import mplot3d
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
bone_list=[[21,5],[5,6],[6,7],[7,8],[8,22],[8,23]]
bone_list=np.array(bone_list)-1
from PIL import Image

import os

def croppingimage(fol1):
#     directory = '/home/paras/Downloads/nturgb/'+str(fol1)
    


    i = Image.open(fol1)
    x1=0
    x2=0
    y1=0
    y2=0
    pixels = i.load() # this is not a list, nor is it list()'able
    width, height = i.size
    print(i.size)
    # print(pixels[1,1])
    all_pixels = []
    f=0
    w=-1
    h=-1
    prev =(255,255,255,255)
    for y in range(height):
        for x in range(width):
            cpixel = pixels[x, y]
            if(cpixel!=(255,255,255,255) and f==0):
    #             print(cpixel)
                f=1
    #             x1=x
                y1=y
            all_pixels.append(cpixel)
            if(cpixel==(255,255,255,255) and prev!=(255,255,255,255)):
    #             x2=w
                y2=h
    #             print(y2,x2)
            prev =cpixel
            w=x
            h=y


    f=0       
    for x in range(width):
        for y in range(height):
            cpixel = pixels[x, y]
            if(cpixel!=(255,255,255,255) and f==0):
    #             print(cpixel)
                f=1
                x1=x
    #             y1=y
            all_pixels.append(cpixel)
            if(cpixel==(255,255,255,255) and prev!=(255,255,255,255)):
                x2=w
    #             y2=h
    #             print(y2,x2)
            prev =cpixel
            w=x
            h=y        
    print(x1,y1,x2,y2) 



    # create a cropped image
    cropped = i.crop((x1,y1,x2,y2))
    # print(pixels[1272,315])
    cropped.save(fol1)
        # show cropped image
#         cropped.show()
        # print(all_pixels)
        





def func(file):
    colors=['b','k','r','g','y','c','m','w']
    fig=plt.figure()
    fig.set_size_inches(48.5,40.5)
#     fig1=plt.figure()
#     fig1.set_size_inches(28.5,20.5)
    ax=fig.add_subplot(111,projection='3d')
#     ax.set(xlim=(-1,2),ylim=(-2,2),zlim=(-1,0.75));
    ax.set(xlim=(-1,2),ylim=(-62,62),zlim=(-1,0.75));
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax1=fig.add_subplot(111,projection='3d')
#     ax.set(xlim=(-1,2),ylim=(-2,2),zlim=(-1,0.75));
    ax1.set(xlim=(-1,2),ylim=(-62,62),zlim=(-1,0.75));
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    plt.axis('off')
    n=0
    c=0
    g=0
    jj=0
    k=1
    x=[]
    y=[]
    z=[]
    n=1
    i=0
    im=0
    z1=0
    w=0
    if(os.path.exists(os.path.join('/home/ws2/Downloads/nturgb+d_skeletons',file))):
        with open(os.path.join('/home/ws2/Downloads/nturgb+d_skeletons',file),'r') as f:
#             print(file)
            
            title=file[:-9]
            print(title)
            for line in f:
                
                    
                
                words=line.split()
                if(i==1 and int(words[0])==0):
                    print('p')
                    return
                if(i==1 and int(words[0])==1):
                    f=1
                    
                else:
                    f=2
                    
#                 print(words)
                
#                 if(i==0):
#                     fr=words[0]
#                     jj=jj+4
#                     i=i+1
#                     continue
                if(f==1):    
                    if(im==25 or k!=1):
                        x.append(float(words[0]))
                        y.append(float(words[1]))
                        z.append(float(words[2]))

    #                     print(jj)
                        jj=jj+1
                        if(k<25):
                            k=k+1
                        else:
                            x=np.array(x)
                            y=np.array(y)
                            z=np.array(z)
    #                         print(x)
                            l=n%8
                            j=colors[l]
                            for bone in bone_list:
                                ax.plot([x[bone[0]],x[bone[1]]],[z[bone[0]]+z1,z[bone[1]]+z1],[y[bone[0]],y[bone[1]]],j)
                            jj=jj+3
                            n=n+1

                            z1=z1+2

                            k=1
                            x=[]
                            y=[]
                            z=[]
                elif(f==2):
                    if((w%2)==0):
                        if(im==25 or k!=1):
                            x.append(float(words[0]))
                            y.append(float(words[1]))
                            z.append(float(words[2]))

        #                     print(jj)
                            jj=jj+1
                            if(k<25):
                                k=k+1
                            else:
                                x=np.array(x)
                                y=np.array(y)
                                z=np.array(z)
        #                         print(x)
                                l=n%8
                                j=colors[l]
                                for bone in bone_list:
                                    ax.plot([x[bone[0]],x[bone[1]]],[z[bone[0]]+z1,z[bone[1]]+z1],[y[bone[0]],y[bone[1]]],j)
                                jj=jj+3
                                n=n+1

                                z1=z1+2
                                w=w+1
                                k=1
                                x=[]
                                y=[]
                                z=[]
                    else:
                        if(im==25 or k!=1):
                            x.append(float(words[0]))
                            y.append(float(words[1]))
                            z.append(float(words[2]))

        #                     print(jj)
                            jj=jj+1
                            if(k<25):
                                k=k+1
                            else:
                                x=np.array(x)
                                y=np.array(y)
                                z=np.array(z)
        #                         print(x)
                                l=n%8
                                j=colors[l]
                                for bone in bone_list:
                                    ax1.plot([x[bone[0]],x[bone[1]]],[z[bone[0]]+z1,z[bone[1]]+z1],[y[bone[0]],y[bone[1]]],j)
                                jj=jj+3
                                n=n+1

                                z1=z1+2
                                w=w+1
                                k=1
                                x=[]
                                y=[]
                                z=[]
                        
                    
                    
                i=i+1
                im=float(words[0])        
        
        plt.savefig('/home/ws2/Documents/nturgb_lefthand/'+title+'.png',dpi=100)
        croppingimage('/home/ws2/Documents/nturgb_lefthand/'+title+'.png')
#         plt.show()

                    
                    
                    
                    
                    
                    
                    
                    
for file in os.listdir('/home/ws2/Downloads/nturgb+d_skeletons'):
    if('P001' in file):
        title=file[:-9]
        if(os.path.exists(os.path.join('/home/ws2/Documents/nturgb_lefthand',title+'.png'))):
            print('continue')
            continue
        func(file)
        
    elif('P001' not in file):
        title=file[:-9]
        if(os.path.exists(os.path.join('/home/ws2/Documents/nturgb_lefthand',title+'.png'))):
            os.remove(os.path.join('/home/ws2/Documents/nturgb_lefthand',title+'.png'))
            print('remove')
    


# In[ ]:




