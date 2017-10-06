#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:56:38 2017
inbreastv2
@author: edu
"""
import sys
sys.path.append("/home/eduardo/research/research/")
import os
import glob
import pickle as pkl
import auxiliary_funcs as auxilf
import funcs.xml_reader as xmlr
from matplotlib import pyplot as plt

src_folder = "/media/eduardo/TOSHIBA EXT/INESC/INbreast/"
set_masses_size = {"train":34,"validation":8,"test":8}
set_neg_size = {"train":40,"validation":9,"test":9}


def get_patients():
    path = src_folder+"AllDICOMs/*.dcm"
    files = glob.glob(path)
    patients = set()
    for file in files:
        patient = os.path.basename(file).split("_")[1]
        patients.add(patient)
        
    return list(patients)

def patient_has_mass(patient):
    files = get_img_files(patient)
    for file in files:
        if xmlr.has_mass(get_mask_files(file)):
            return True
    return False

def get_img_files(patient):
    path = src_folder+"AllDICOMs/*"+patient+"*.dcm"
    files = glob.glob(path)
    return files

def get_mask_files(file):
    image_number = os.path.basename(file).split("_")[0]
    file = src_folder+"AllXML/"+image_number+".xml"
    return file
    
def load_mask(m_file, original_shape):
    return xmlr.get_mask_images(m_file,original_shape)
    
    
def add_images(dataset,split,patient,folder,number):
    files = get_img_files(patient)
    
    for file in files:
        img = auxilf.read_dicom_file(file)
        ori_shape = img.shape
        img,roi = auxilf.preprocess(img)
        file_name = auxilf.save_image(img,folder,number,"img")
        dataset[split].append(file_name)
        auxilf.save_image(roi,folder,number,"roi")
        
        m_file = get_mask_files(file)
        masks = load_mask(m_file,ori_shape)
        
        suffix=0
        for mask in masks:
            mask = auxilf.preprocess_binary(mask,img.shape)
            auxilf.save_image(mask,folder,number,"mask",suffix)
            suffix+=1
        number+=1
        
        if False:
            f, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(img)
            plt.axis('off')
            ax2.imshow(roi)
            plt.axis('off')
            if suffix>0:
                ax3.imshow(masks[0])
            plt.axis('off')
            plt.show()

    return number

    
def create_standard_format(folder):
    dataset = {"train":[],"validation":[],"test":[]}
    patients = get_patients()
    
    os.mkdir(folder)
    
    with_masses=list()
    no_masses=list()
    for patient in patients:
        if patient_has_mass(patient):
            with_masses.append(patient)
        else:
            no_masses.append(patient)
    number = 0
    for split in set_masses_size.keys():
        for i in range(set_masses_size[split]):
            number = add_images(dataset,split,with_masses.pop(),folder,number)
            
    for split in set_neg_size.keys():
        for i in range(set_neg_size[split]):
            number = add_images(dataset,split,no_masses.pop(),folder,number)
    
    pkl.dump(dataset,open(folder+"/data","wb"))
            
create_standard_format("/home/eduardo/inbreast_full_images_12")
    
    
    
    
