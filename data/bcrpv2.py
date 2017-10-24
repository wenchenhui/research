# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:48:52 2017
bcrp
@author: eduardo
"""
import sys
sys.path.append("/home/eduardo/research/research/")
import glob
import os
from scipy import misc
import auxiliary_funcs as auxilf
from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np

def get_folders(set_="train"):
    folders = []
    if set_ == "train" or set_ == "validation":
        file = open("/data/DB/ddsm/bcrp/BCRP_MASS_0_train.txt","r")
    if set_ == "test":
        file = open("/data/DB/ddsm/bcrp/BCRP_MASS_1_test.txt","r")
    init_path = "/data/DB/ddsm/ALL/cancers/"
    
    content = file.readlines()
    file.close()
        
    for line in content:
        folders.append(init_path+line[:-1])
    return folders
    
def get_images_and_masks(folder):
    files = glob.glob(folder+"/*.png")
    images = []
    for file in files:
        if file[-8::] == "mask.png":
            continue
        images.append(file)
        
    masks = dict()
    for image in images:
        masks[image] = glob.glob(image[:-4]+".annot*.png")
    
    return images, masks

def add_images(dataset,split,folder,dst_directory,number):
    
    image_files,mask_files = get_images_and_masks(folder)
    
    for file in image_files:
        img = misc.imread(file).astype(np.uint16)
        plt.imshow(img)
        plt.show()
        img,roi = auxilf.preprocess(img)
        file_name = auxilf.save_image(img,dst_directory,number,"img")
        dataset[split].append(file_name)
        auxilf.save_image(roi,dst_directory,number,"roi")
        
        masks_local = mask_files[file]
        
        suffix=0
        for mask_file in masks_local:
            mask = misc.imread(mask_file)
            mask = auxilf.preprocess_binary(mask,img.shape)
            auxilf.save_image(mask,dst_directory,number,"mask",suffix)
            suffix+=1
        number+=1
        
        if True:
            f, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(img)
            plt.axis('off')
            ax2.imshow(roi)
            plt.axis('off')
            if suffix>0:
                ax3.imshow(mask)
            plt.axis('off')
            plt.show()

    return number

def create_standard_format(dst_directory):

    folders = get_folders("train")
    folders_dict = dict()
    folders_dict["validation"] = folders[0:9]
    folders_dict["train"] = folders[9::]
    folders_dict["test"] = get_folders(set_="test")
    
    dataset = {"train":[],"validation":[],"test":[]}
    os.mkdir(dst_directory)
    
    number = 0
    for split in ["train","validation","test"]:
        for folder in folders_dict[split]:
            number = add_images(dataset,split,folder,dst_directory,number)
    
    pkl.dump(dataset,open(dst_directory+"/data","wb"))    

#create_standard_format("/home/eduardo/bcrp_full_images_12/")