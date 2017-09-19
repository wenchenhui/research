# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:09:06 2017

@author: eduardo

CBIS Standart data format
"""

import glob
import os
import utils as ut
import dicom
import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage.morphology as mo
import skimage.measure as me


src_location = "/media/edu/TOSHIBA EXT/"
folder_images = "raw CBIS mass/DOI/"
folder_rois = "raw CBIS mass rois/DOI/"
dst_location = "/media/edu/TOSHIBA EXT/dataset2/"

def list_of_patients():
    image_folders = glob.glob(os.path.join(src_location,folder_images,"*"))
    patients = set()

    for file in image_folders:
        patient = file.split("_")[2]
        patients.add(patient)
    
    return list(patients)

def images_of_pat(patient):
    folders = glob.glob(os.path.join(src_location,folder_images,"*_"+patient+"_*"))
    images = []
    for folder in folders:
        first_folder = glob.glob(folder+"/*")[0]
        second_folder = glob.glob(first_folder+"/*")[0]
        file = glob.glob(second_folder+"/*.dcm")[0]
        images.append(file)
        
    return images
    
def masks_of_image(file):
    folder = file.split("/")[-4]
    folders = glob.glob(os.path.join(src_location,folder_rois,folder+"_*"))

    masks = [] 
    for folder in folders:
        files_list = []

        for path, subdirs, files in os.walk(folder):
            for name in files:
                files_list.append(os.path.join(path, name))
                
        file = get_files_by_file_size(files_list)[-1]
        masks.append(file)
        
    return masks
    
scale = 1/24
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))    
def load_preprocess_save_image(file,patient,image_ids):
    
    #Load image
    img = dicom.read_file(file).pixel_array
    
    #Stretch histogram
    M = img.max()
    m = img.min()
    img = (img-m)/(M-m)

    #Resize image
    img = cv2.resize(img,(0,0),fx = scale,fy = scale)
    
    #Remove artifacts
    roi,img = get_roi(img)
    
    #Save image
    np.save(dst_location+patient+"_img_"+str(image_ids),img)
    np.save(dst_location+patient+"_roi_"+str(image_ids),roi)
    #print(img.max(),img.min())
    plt.imshow(img)
    plt.show()
    plt.imshow(roi)
    plt.show()
    return roi


def get_roi(img,disk_size = int(60*4/24)):
    img = np.pad(img,disk_size+1,"constant")
    hat = mo.white_tophat(img,selem = mo.disk(disk_size))
    pre = img-hat
    mask = pre>=0.01
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
    
def load_preprocess_save_mask(file,patient,image_ids,mask_ids,roi):
    #mask = dicom.read_file(file).pixel_array
    
    try:
        mask = dicom.read_file(file).pixel_array
    except:
        aux = dicom.read_file(file)
        aux[65536*32736+16].value=aux[65536*32736+16].value[1::]
        mask = aux.pixel_array
        #print("first: ",aux[65536*32736+16].value[0],", second: ",aux[65536*32736+16].value[-1])
            
    mask = cv2.resize(mask,roi.shape[::-1])
    plt.imshow(mask)
    plt.show()
    print(np.sum(mask*roi)/np.sum(mask))
    assert np.sum(mask*roi) > np.sum(mask)*0.5
    np.save(dst_location+patient+"_mask_"+str(image_ids)+"_"+str(mask_ids),mask)
    
    return
    
def make_CBIS_standart_format():
    
    print("SAVING CBIS IN THE STANDART FORMAT. THANK YOU FOR PROCESSING WITH US!")
    os.makedirs(dst_location)
    patients = list_of_patients()
    #patient_dict = dict()
    
    image_ids = 0

    bar = ut.progress_bar(len(patients))
    for patient in patients:
        
        image_files = images_of_pat(patient)
        #patient_dict[patient] = image_files

        for file in image_files:
            roi = load_preprocess_save_image(file,patient,image_ids)
            masks_files = masks_of_image(file)

            mask_ids = 0
            for file2 in masks_files:
                print(file2)
                load_preprocess_save_mask(file2, patient,image_ids, mask_ids, roi)
                mask_ids+=1

            image_ids+=1
        
        bar.tick()
        
    print("WAIT IS OVER BOY! SEE YOU NEXT MONTH.")
        
        
def get_files_by_file_size(files, reverse=False):
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
TEST NUMBER OF PATIENTS AND IMAGES
"""

def print_number_of_patients():
    pats = list_of_patients()
    print(len(pats),"patients")
    images = 0
    for i in range(len(pats)):
        images+=len(images_of_pat(pats[i]))
    print(images,"images")

make_CBIS_standart_format()

