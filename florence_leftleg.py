#!/usr/bin/env python
# coding: utf-8

# In[1]:





import os
from mpl_toolkits import mplot3d
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


from PIL import Image

import os
bone_list=[[2,9],[9,10],[10,11]]
bone_list=np.array(bone_list)-1
def croppingimage(fol1):
#     directory = '/home/paras/Downloads/nturgb/'+str(fol1)
    


    i = Image.open(fol1)
    x1=0
    y1=0
    x2=0
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


def func():
    colors=['b','k','r','g','y','c','m','w']
    fig=plt.figure()
    fig.set_size_inches(68.5,60.5)
#     fig1=plt.figure()
#     fig1.set_size_inches(28.5,20.5)
    ax=fig.add_subplot(111,projection='3d')
#     ax.set(xlim=(-1,2),ylim=(-2,2),zlim=(-1,0.75));
    ax.set(xlim=(-500,500),ylim=(1500,4500),zlim=(-1000,1000));
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
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
    v=1
    p=1
    a=1
    
    if(os.path.exists(os.path.join('/home/ws2/Downloads/Florence_3d_actions','Florence_dataset_WorldCoordinates.txt'))):
        with open(os.path.join('/home/ws2/Downloads/Florence_3d_actions','Florence_dataset_WorldCoordinates.txt'),'r') as f:
            for line in f:
                words=line.split()
                if(os.path.exists('/home/ws2/Documents/flore_leftleg/'+'a'+words[2]+'s'+words[0]+'p'+words[1]+'.png')):
                    print('continue')
                    continue
                if(int(words[0]) != v or int(words[1])!= p or int(words[2])!=a ):
                    f=1
                elif(int(words[0]) == v and int(words[1])== p and int(words[2])==a ):
                    f=2
                    
                    
                if(f==2):
                    
                    
                    for i in range(3,48,3):
                        x.append(float(words[i]))
                        i=i+1
                        y.append(float(words[i]))
                        i=i+1
                        z.append(float(words[i]))
                        i=i+1

    #                     print(jj)
                        
                    x=np.array(x)
                    y=np.array(y)
                    z=np.array(z)                        
#                     print(x)
#                     print(y)
#                     print(z)
                    l=n%8
                    j=colors[l]
                    for bone in bone_list:
                        ax.plot([x[bone[0]],x[bone[1]]],[z[bone[0]]+z1,z[bone[1]]+z1],[y[bone[0]],y[bone[1]]],j)

                    n=n+1
                    
#                     ax.view_init(30, )
#                     plt.show()
                    
#                     return
                    z1=z1+2
                    k=1
                    x=[]
                    y=[]
                    z=[]
#                     return
                elif(f==1):
#                     return
                    print(int(words[2]))
#                     plt.show()
                    plt.savefig('/home/ws2/Documents/flore_leftleg/'+'a'+words[2]+'s'+words[0]+'p'+words[1]+'.png',dpi=100)
                    croppingimage('/home/ws2/Documents/flore_leftleg/'+'a'+words[2]+'s'+words[0]+'p'+words[1]+'.png')
#                     return
                    z1=0
                    fig=plt.figure()
                    fig.set_size_inches(68.5,60.5)
                 #     fig1=plt.figure()
                 #     fig1.set_size_inches(28.5,20.5)
                    ax=fig.add_subplot(111,projection='3d')
                 #     ax.set(xlim=(-1,2),ylim=(-2,2),zlim=(-1,0.75));
                    ax.set(xlim=(-500,500),ylim=(1500,4500),zlim=(-1000,1000));
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_zlabel('z')
                    ax.grid(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_zticks([])

                    plt.axis('off')
                    v=int(words[0])
                    p=int(words[1])
                    a=int(words[2])
                    
                        
                    
#     plt.show()                
    plt.savefig('/home/ws2/Documents/flore_leftleg/'+'a'+words[2]+'s'+words[0]+'p'+words[1]+'.png',dpi=100)
    croppingimage('/home/ws2/Documents/flore_leftleg/'+'a'+words[2]+'s'+words[0]+'p'+words[1]+'.png')                    
        
func()       


# In[ ]:




