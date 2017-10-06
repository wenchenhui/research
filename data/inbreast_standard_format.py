# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 17:28:25 2017

@author: eduardo
"""
import sys
sys.path.append("/home/eduardo/research/research/")
import glob
import os
import dicom
import funcs.xml_reader as xmlr
import funcs.utils as ut
import cv2 
import numpy as np
from scipy.misc import imsave
from matplotlib import pyplot as plt



src_folder = "/media/eduardo/TOSHIBA EXT/INESC/INbreast/"

def _list_of_patients():
    path = src_folder+"AllDICOMs/*.dcm"
    files = glob.glob(path)
    patients = set()
    for file in files:
        patient = os.path.basename(file).split("_")[1]
        patients.add(patient)
        
    return list(patients)
        
    
    
def _images_of_pat(patient):
    path = src_folder+"AllDICOMs/*"+patient+"*.dcm"
    files = glob.glob(path)
    return files

def _masks_of_image(file):
    image_number = os.path.basename(file).split("_")[0]
    file = src_folder+"AllXML/"+image_number+".xml"
    return file

_clahe_op = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
def _load_preprocess_save_image(file,patient,image_ids):
    #Load image
    img = dicom.read_file(file).pixel_array
    original_shape = img.shape
    if _clahe:
        img = _clahe_op.apply(img)    
        
    M = img.max()
    m = img.min()
    img2 = ((img-m)/(M-m))*255
    img2 = img2.astype(np.uint8)
    
    img = cv2.resize(img2,(0,0),fx = _scale,fy = _scale,interpolation=cv2.INTER_AREA)        
    
    # Remove artifacts
    roi = _get_roi(img)
    img = (img.astype(float)-128)/256
    #plt.imshow(img)
    #plt.show()
    #input("next?")
    # Save image
    if _debug:
        imsave(_dst_location+patient+"_img_"+str(image_ids)+".png",img)
        imsave(_dst_location+patient+"_roi_"+str(image_ids)+".png",roi)
    else:
        np.save(_dst_location+patient+"_img_"+str(image_ids),img)
        np.save(_dst_location+patient+"_roi_"+str(image_ids),roi)
    
    return roi,original_shape


def _get_roi(img):
    return img>5

def _load_preprocess_save_masks(file,patient,image_ids,roi,original_shape):
    
    if not os.path.isfile(file): return
        
    masks = xmlr.get_mask_images(file,original_shape)
    mask_ids=0
    for mask in masks:
        mask = cv2.resize(mask.astype(float),roi.shape[::-1])
        mask = mask>0
        #plt.imshow(mask)
        #plt.show()
        assert np.sum(mask*roi) > np.sum(mask)*0.5
    
        if _debug:
            imsave(_dst_location+patient+"_mask_"+str(image_ids)+"_"+str(mask_ids)+".png",mask)
        else:
            np.save(_dst_location+patient+"_mask_"+str(image_ids)+"_"+str(mask_ids),mask)
            #print("mask", mask.max())
        mask_ids+=1
    return



def make_INbreast_standart_format(inv_scale, clahe, dst_location, debug):
    global _scale,_clahe,_dst_location,_debug
    _scale,_clahe,_dst_location,_debug = 1/inv_scale, clahe, dst_location, debug
    
    
    print("TRANSFORMING INbreast TO STANDARD FORMAT")
    os.makedirs(_dst_location)
    
    patients = _list_of_patients()
    bar = ut.progress_bar(len(patients))
    image_ids=0
    for patient in patients:
        bar.tick()
        
        images = _images_of_pat(patient)
        
        for image in images:
            
            roi,original_shape = _load_preprocess_save_image(image,patient,image_ids)
            masks = _masks_of_image(image)
            
            _load_preprocess_save_masks(masks, patient,image_ids, roi,original_shape)
            image_ids+=1
    
def check_with_mass_vs_healthy(src_path):
    with_masses = 0
    no_masses = 0
    n_images = len(glob.glob(src_path+"/*_img_*.npy"))
    
    for iD in range(n_images):
        paths = glob.glob(src_path+"/*_img_"+str(iD)+".npy")
        assert len(paths) == 1
        if len(glob.glob(src_path+"/*_mask_"+str(iD)+"_*")) > 0:
            with_masses += 1
        else: 
            no_masses += 1
    print(with_masses)
    print(no_masses)
        
        
    
    
    
    
    
    
    
    