# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:24:17 2017

@author: eduardo
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import dicom
import glob
import cv2

clahe_op = cv2.createCLAHE(clipLimit=4, tileGridSize=(8,8))

def show_histogram(img,bins = 256,inter=[0,256]):
    plt.hist(img.ravel(),bins,inter)
    plt.axis('off')

for file in glob.glob("/media/eduardo/TOSHIBA EXT/INESC/INbreast/AllDICOMs/*.dcm"):
    img = dicom.read_file(file).pixel_array
    
    img = clahe_op.apply(img)    
    M = img.max()
    m = img.min()
    img2 = ((img-m)/(M-m))*255
    img2 = img2.astype(np.uint8)
    img2 = cv2.resize(img2,(0,0),fx = 1/12,fy = 1/12,interpolation=cv2.INTER_AREA)    
    
    
    fig, ax = plt.subplots(nrows=2, ncols=2)    
    fig.set_figheight(15)
    fig.set_figwidth(15)
     
    plt.subplot(2,3,1)    
    show_histogram(img,inter=[1,img.max()])
    
    
    plt.subplot(2,3,4)    
    plt.imshow(img)
    plt.axis('off') 
       
       
    
    plt.subplot(2,3,2)    
    show_histogram(img,inter=[1,img.max()])
    
    plt.subplot(2,3,5) 
    plt.imshow(img)
    plt.axis('off')    
    
    #img2 = clahe_op.apply(img2)    
    
    plt.subplot(2,3,3)    
    show_histogram(img2,inter=[1,img.max()])

    plt.subplot(2,3,6) 
    plt.imshow(img2)
    plt.axis('off')
    
    
    plt.show()
    input("next?")
    