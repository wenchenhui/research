# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:09:06 2017

@author: eduardo

CBIS Standart data format

References:
    [1] -   Rebecca Sawyer Lee, Francisco Gimenez, Assaf Hoogi , Daniel Rubin
            (2016). Curated Breast Imaging Subset of DDSM. The Cancer Imaging
            Archive. http://dx.doi.org/10.7937/K9/TCIA.2016.7O02S9CY
            
    [2] -   Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization.
            Graphics Gems, 474-485. doi:10.1016/b978-0-12-336156-1.50061-6
"""

import glob
import os
import funcs.utils as ut
import dicom
import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage.morphology as mo
import skimage.measure as me
from scipy.misc import imsave


"""
_______________________________________________________________________________
___________________________________AZEALIA_____________________________________
FUNCTIONS TO LIST DATA:
    GET ALL PATIENTS, GET IMAGES OF EACH PATIENT, GET MASKS OF EACH IMAGE
"""
def _list_of_patients():
    image_folders = glob.glob(os.path.join(_src_location,_folder_images,"*"))
    patients = set()

    for file in image_folders:
        patient = file.split("_")[2]
        patients.add(patient)
    
    return list(patients)

def _images_of_pat(patient):
    folders = glob.glob(os.path.join(_src_location,_folder_images,"*_"+patient+"_*"))
    images = []
    for folder in folders:
        first_folder = glob.glob(folder+"/*")[0]
        second_folder = glob.glob(first_folder+"/*")[0]
        file = glob.glob(second_folder+"/*.dcm")[0]
        images.append(file)
        
    return images
    
def _masks_of_image(file):
    folder = file.split("/")[-4]
    folders = glob.glob(os.path.join(_src_location,_folder_masks,folder+"_*"))

    masks = [] 
    for folder in folders:
        files_list = []

        for path, subdirs, files in os.walk(folder):
            for name in files:
                files_list.append(os.path.join(path, name))
                
        file = _get_files_by_file_size(files_list)[-1]
        masks.append(file)
        
    return masks

# AUXILIARY    
def _get_files_by_file_size(files, reverse=False):
    """ Return list of file paths in directory sorted by file size """
    # Re-populate list with filename, size tuples
    for i in range(len(files)):
        files[i] = (files[i], os.path.getsize(files[i]))

    # Sort list by file size
    # If reverse=True sort from largest to smallest
    # If reverse=False sort from smallest to largest
    files.sort(key=lambda filename: filename[1], reverse=reverse)

    # Re-populate list with just filenames
    for i in range(len(files)):
        files[i] = files[i][0]

    return files




"""
_______________________________________________________________________________
____________________________________BANKS______________________________________
FUNCTIONS TO PROCESS IMAGES:
    LOAD/PREPROCESS/SAVE IMAGES/ROIS/MASKS
"""

_clahe_op = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
def _load_preprocess_save_image(file,patient,image_ids):
    
    #Load image
    img = dicom.read_file(file).pixel_array
    
    #Stretch histogram and convert to uint8
    M = img.max()
    m = img.min()
    img = ((img-m)/(M-m))*255
    img = img.astype(np.uint8)
    
    # Apply CLAHE method [2]
    if _clahe:
        img = _clahe_op.apply(img)
    
    # Resize image
    img = cv2.resize(img,(0,0),fx = _scale,fy = _scale)
    
    # Remove artifacts
    roi,img = _get_roi(img)
    
    # Save image
    if _debug:
        imsave(_dst_location+patient+"_img_"+str(image_ids)+".png",img)
        imsave(_dst_location+patient+"_roi_"+str(image_ids)+".png",roi)
    else:
        np.save(_dst_location+patient+"_img_"+str(image_ids),img.astype(float)/255)
        np.save(_dst_location+patient+"_roi_"+str(image_ids),roi)
    
    return roi

def _get_roi(img,disk_size = int(60*4/24)):
    img = np.pad(img,disk_size+1,"reflect")
    hat = mo.white_tophat(img,selem = mo.disk(disk_size))
    pre = img-hat
    mask = pre>=10
    #plt.imshow(mask)
    #plt.show()
    #plt.imshow(img)
    #plt.show()
    labels = me.label(mask,connectivity=2)
    if labels.max()>1:
        #print("one escaped")
        reg = me.regionprops(labels)
        selected = 0
        for i in range(len(reg)):
            if reg[i].area > reg[selected].area:
                selected=i
        mask = (labels==reg[selected].label)
        
    mask = mo.dilation(mask,selem=mo.disk(int(disk_size/2)))
    final = img*mask
    rm = disk_size+1
    
    return mask[rm:-rm,rm:-rm],final[rm:-rm,rm:-rm]
    
def _load_preprocess_save_mask(file,patient,image_ids,mask_ids,roi):
    
    try:
        mask = dicom.read_file(file).pixel_array
    except:
        aux = dicom.read_file(file)
        aux[65536*32736+16].value=aux[65536*32736+16].value[1::]
        mask = aux.pixel_array
        #print("first: ",aux[65536*32736+16].value[0],", second: ",aux[65536*32736+16].value[-1])
            
    mask = cv2.resize(mask,roi.shape[::-1])
    mask = mask>0
    #plt.imshow(mask)
    #plt.show()
    assert np.sum(mask*roi) > np.sum(mask)*0.5
    
    if _debug:
        imsave(_dst_location+patient+"_mask_"+str(image_ids)+"_"+str(mask_ids)+".png",mask)
    else:
        np.save(_dst_location+patient+"_mask_"+str(image_ids)+"_"+str(mask_ids),mask)
    #print("mask", mask.max())
    return
    
    
"""
_______________________________________________________________________________
________________________________CLEMENTINE_____________________________________
MAIN FUNCTION
    CREATES THE STANDARD DATASET
"""
def make_CBIS_standart_format(scale, clahe, src_location, folder_images, folder_masks, dst_location, debug):
    
    # DEFINE SETTINGS AS GLOBAL VARIABLES
    global _scale, _clahe, _src_location, _folder_images, _folder_masks, _dst_location, _debug
    _scale = scale
    _clahe = clahe
    _src_location = src_location
    _folder_images = folder_images
    _folder_masks = folder_masks
    _dst_location = dst_location
    _debug = debug
    
    # PRINT INFORMATION ABOUT THE DATASET
    print("TRANSFORMING CBIS TO STANDARD FORMAT")
    os.makedirs(dst_location)
    no_pats,no_imgs,no_masks = _number_of_patients_and_images()
    print("\t Processing ",no_pats," patients")
    print("\t Processing ",no_imgs," images")
    print("\t Processing ",no_masks," masks")
    
    # PROGRESS BAR OBJECT
    bar = ut.progress_bar(no_pats)
    
    # GET AVAILABLE PATIENTS
    patients = _list_of_patients()
    
    # IMAGE ID TRACKER (MAKES SURE NO TWO IMAGES HAVE THE SAME NAME)
    image_ids = 0

    # WE PROCESS ONE PATIENT AT A TIME
    for patient in patients:
        image_files = _images_of_pat(patient)
        
        # WE PROCESS ONE IMAGE AT A TIME
        for file in image_files:
            
            # SAVE THIS IMAGE AND ROI AS A NPY/PNG
            # ROI IS RETURNED TO ASSERT EVERY MASK IS INSIDE THE REGION OF INTEREST
            roi = _load_preprocess_save_image(file,patient,image_ids)
            
            #GET MASKS
            masks_files = _masks_of_image(file)
            
            # MASK ID TRACKER (IT IDENTIFIES EACH MASK IN THE SAME IMAGE)
            """ WARNING: IDS CAN BE DIFFERENT FROM THE ONES FOUND IN THE CBIS 
            CSV FILES"""
            mask_ids = 0
            
            # WE PROCESS EACH MASK INDIVIDUALLY
            for file2 in masks_files:
                
                # SAVE THIS MASK AS A NPY/PNG
                _load_preprocess_save_mask(file2, patient,image_ids, mask_ids, roi)
                mask_ids+=1

            image_ids+=1
        
        # PROGRESS BAR UPDATE
        bar.tick()
        
    print("A NEW CBIS DATASET IN THE STANDART FORMAT WAS SAVED IN:",_dst_location)
    print("Parameters:")
    print("\t->scale",_scale)
    print("\t->clahe",_clahe)


# AUXILIARY    
def _number_of_patients_and_images():
    pats = _list_of_patients()
    #print(len(pats),"patients")
    
    no_images = 0
    no_masks = 0
    for i in range(len(pats)):
        images = _images_of_pat(pats[i])
        no_images+=len(images)
        
        for j in range(len(images)):
            masks = _masks_of_image(images[j])
            no_masks+=len(masks)
    return len(pats),no_images,no_masks

