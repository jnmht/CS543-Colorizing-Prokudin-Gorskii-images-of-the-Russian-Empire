

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

for name in l:
    imname=name
    input_img=Image.open(imname)
    input_img=np.asarray(input_img)

    

    w,h=input_img.shape
    height=int(w/3)
    b_lue=input_img[0:int(height),:]
    gre_en=input_img[int(height):2*int(height),:]
    re_d=input_img[2*int(height):3*int(height),:]
    
   

    final_image = (np.dstack((re_d,gre_en,b_lue))).astype(np.uint8)

    
    
    final_image = Image.fromarray(final_image)
    final_image.save("op/"+name)
    plt.figure()
    plt.imshow(final_image)
    

