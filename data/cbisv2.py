# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 19:22:38 2017
cbisv2
@author: eduardo
"""
import sys
sys.path.append("/home/eduardo/research/research/")
import os
import glob
import pickle as pkl
import auxiliary_funcs as auxilf
import funcs.xml_reader as xmlr
from matplotlib import pyplot as plt


src_folder = "/media/eduardo/TOSHIBA EXT/CBIS_DATASET/"
def get_patients(set_ = "train"):
    if set_ == "train" or set_ == "validation":
        path = src_folder+"Images/DOI/*"
    elif set_ == "test":
        path = src_folder+"Test Images/DOI/*"
    else:
        raise(Exception("invalid input: set_"))
    files = glob.glob(path)
    patients = set()
    for file in files:
        patient = os.path.basename(file).split("_")[2]
        patients.add(patient)
        
    return list(patients)

def get_img_folders(set_, patient):
    if set_ == "train" or set_ == "validation":
        path = src_folder+"Images/DOI/"+"Mass-Training_P_"+patient+"*"
    elif set_ == "test":
        path = src_folder+"Test Images/DOI/"+"Mass-Test_P_"+patient+"*"
    else:
        raise(Exception("invalid input: set_"))

    folders = glob.glob(path)
    return folders

def get_masks_folders(set_,img_folder):
    if set_ == "train" or set_ == "validation":
        path = src_folder+"Masks/DOI/"+os.path.basename(img_folder)+"_*"
    elif set_ == "test":
        path = src_folder+"Test Masks/DOI/"+os.path.basename(img_folder)+"_*"
    else:
        raise(Exception("invalid input: set_"))

    folders = glob.glob(path)
    return folders
    
def get_file_in_folder(original_folder):
    files = []
    first_folders = glob.glob(original_folder+"/*")
    second_folders = []
    for folder in first_folders:
        second_folders += glob.glob(folder+"/*")
    for folder in second_folders:
        files += glob.glob(folder+"/*")
        
    return sorted(files,key = os.path.getsize)[-1]

def add_images(dataset,split,patient,dst_directory,number):
    folders = get_img_folders(split,patient)
    
    for img_folder in folders:
        img = auxilf.read_dicom_file(get_file_in_folder(img_folder))
        img,roi = auxilf.preprocess(img)
        file_name = auxilf.save_image(img,dst_directory,number,"img")
        dataset[split].append(file_name)
        auxilf.save_image(roi,dst_directory,number,"roi")
        
        masks = get_masks_folders(split,img_folder)
        
        suffix=0
        for mask_folder in masks:
            mask = auxilf.read_dicom_file(get_file_in_folder(mask_folder),allow_wrong_sizes=True)
            mask = auxilf.preprocess_binary(mask,img.shape)
            auxilf.save_image(mask,dst_directory,number,"mask",suffix)
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

    
def create_standard_format(dst_directory):

    patients = get_patients(set_="train")
    pat_dict = dict()
    pat_dict["validation"] = patients[0:104]
    pat_dict["train"] = patients[104::]
    pat_dict["test"] = get_patients(set_="test")
    
    dataset = {"train":[],"validation":[],"test":[]}
    os.mkdir(dst_directory)
    
    number = 0
    for split in ["train","validation","test"]:
        for patient in pat_dict[split]:
            number = add_images(dataset,split,patient,dst_directory,number)
    
    pkl.dump(dataset,open(dst_directory+"/data","wb"))    

#create_standard_format("/home/eduardo/cbis_full_images_12/")