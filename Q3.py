
# coding: utf-8

# In[13]:


import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import numpy as np
# from scipy.misc import imresize
#vancouver_tableau


l=[]
import os
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    if ".jpg" in f:
        l.append(f)
try: 
    os.mkdir("op") 
except OSError as error: 
    ("")
def ncc(image1,image2):
    image1=image1-image1.mean(axis=0)
    image2=image2-image2.mean(axis=0)
    return np.sum(((image1/np.linalg.norm(image1)) * (image2/np.linalg.norm(image2))))

def nccAlign(image1, image2, ofsetx,ofsety,_range):
    ncc_min = -1
    value=np.linspace(-_range+ofsetx,_range+ofsetx,2*_range,dtype=int)
    for i in value:
        for j in value:
            Diffncc = ncc(image1,np.roll(image2,[i,j],axis=(0,1)))
            if Diffncc > ncc_min:
                ncc_min = Diffncc
                output = [i,j]
#     print(output)
    return output
#'vancouver_tableau.jpg'
for name in l:
    imname=name
    img1=Image.open(imname)
    img=np.asarray(img1)
    w,h=img.shape
    print(w,h)
    plt.figure()
    plt.imshow(img)

    w,h=img.shape

    plt.imshow(img)

    scale=1/2
    # image=imresize(img,scale)
    image=img1.resize((int(h/2), int(w/2)))


    img=np.asarray(image)
    print(img.shape)
    w,h=img.shape
    height=int(w/3)
    b_lue=img[0:height,:]
    gre_en=img[height:2*height,:]
    re_d=img[2*height:3*height,:]
    
    
    
    img1=np.asarray(img1)
    he_=int(img1.shape[0]/3)
    
    b_=img1[0:he_,:]
    g_=img1[he_:2*he_,:]
    r_=img1[2*he_:3*he_,:]
    
    
    xgb,ygb=0,0
    xrb,yrb=0,0
    GtoBalign = nccAlign(b_lue,gre_en,xgb,ygb,15)
    RtoBalign = nccAlign(b_lue,re_d,xrb,yrb,15)
    x_gtob,y_gtob=GtoBalign[0]*2,GtoBalign[1]*2
    x_rtob,y_rtob=RtoBalign[0]*2,RtoBalign[1]*2

    g_=np.roll(g_,[x_gtob,y_gtob],axis=(0,1))
    r_=np.roll(r_,[x_rtob,y_rtob],axis=(0,1))
    print([x_gtob,y_gtob])
    print([x_rtob,y_rtob])

    r_to_b = nccAlign(b_,r_,x_rtob,y_rtob,7)
    g_to_b = nccAlign(b_,g_,x_gtob,y_gtob,7)


    r=np.roll(r_,r_to_b,axis=(0,1))
    g=np.roll(g_,g_to_b,axis=(0,1))

    print(r_to_b)
    print(g_to_b)
    final_image = (np.dstack((r,g,b_))).astype(np.uint8)

    final_image = Image.fromarray(final_image)
    final_image.save("op/"+name)
    plt.figure()
    plt.imshow(final_image)


# In[3]:


g.shape

