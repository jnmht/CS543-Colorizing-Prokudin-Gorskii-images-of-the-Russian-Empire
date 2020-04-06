
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import numpy as np
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
def nccAlign(image1, image2, _range):
    ncc_min = -1
    value=np.linspace(-_range,_range,2*_range,dtype=int)

    for i in value:
        for j in value:
            Diffncc = ncc(image1,np.roll(image2,[i,j],axis=(0,1)))
            if Diffncc > ncc_min:
                ncc_min = Diffncc
                output = [i,j]
    return output
for name in l:
    imname=name
    input_img=Image.open(imname)
    input_img=np.asarray(input_img)



    height=int(np.floor(input_img.shape[0] / 3).astype(np.int))

    b=input_img[0:int(height),:]
    grn=input_img[int(height):2*int(height),:]
    red=input_img[2*int(height):3*int(height),:]
    
    r_to_b = nccAlign(b,red,20)
    g_to_b = nccAlign(b,grn,20)
   
    print(g_to_b, r_to_b)
    r=np.roll(red,r_to_b,axis=(0,1))
    g=np.roll(grn,g_to_b,axis=(0,1))

    final_image = (np.dstack((r,g,b))).astype(np.uint8)

    final_image = Image.fromarray(final_image)
    final_image.save("op/"+name)
    plt.figure()
    plt.imshow(final_image)
    

