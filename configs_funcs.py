# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 17:43:39 2017

@author: eduardo
"""

import data.cbis_standart_format as cbis_standart_format
import data.inbreast_standard_format as inbreast_standard_format
import simple_data_processor
import train_loop
import pickle as pkl
import os



"""
PRE DATA PROCESSING
"""
def get_path_predata(dataset,inv_scale,clahe):
    clahe_text = ""
    if clahe:
        clahe_text += "_clahe"
    return "/home/eduardo/predata_"+dataset+"_"+str(inv_scale)+clahe_text+"/"
        
def create_standart_predata(dataset,inv_scale,clahe,debug,
                            src_location = "/media/eduardo/TOSHIBA EXT/",
                            folder_images = "raw CBIS mass/DOI/",
                            folder_masks = "raw CBIS mass rois/DOI/"):
    
    dst_location = get_path_predata(dataset,inv_scale,clahe)    
    if dataset == "CBIS":        
        cbis_standart_format.make_CBIS_standart_format(1/inv_scale,clahe,src_location,folder_images,folder_masks,dst_location,debug)
    elif dataset == "inbreast":
        inbreast_standard_format.make_INbreast_standart_format(inv_scale,clahe,dst_location,debug)
    
def get_path_data1(dataset,use_elastic_deform,no_transformations):
    use_elastic_deform_text = ""
    if use_elastic_deform:
        use_elastic_deform_text += "_deform_elas"
    return "/home/eduardo/data_"+dataset+"_"+str(no_transformations)+use_elastic_deform_text+"/"
    
def create_first_dataset(src_location, dataset_name, use_elastic_deform, no_transformations):
    
    dst_location = get_path_data1(dataset_name, use_elastic_deform, no_transformations)
    
    if dataset_name == "CBIS":
        set_masses_size = {"train":491,"validation":100,"test":100}
        set_neg_size = {"train":0,"validation":0,"test":0}
    elif dataset_name == "inbreast":
        set_masses_size = {"train":34,"validation":8,"test":8}
        set_neg_size = {"train":40,"validation":9,"test":9}
    elif dataset_name == "inbreast_12":
        set_masses_size = {"train":34,"validation":8,"test":8}
        set_neg_size = {"train":40,"validation":9,"test":9}
    
    
    dataset = simple_data_processor.make_dataset(set_masses_size, set_neg_size, src_location, dst_location)

    # PATCHES
    #halfpatch= 18
    
    use_rotations = True
    use_mirroring = True
    
    debug = False
    dataset.make_patches_dataset(no_transformations,use_rotations,use_mirroring,use_elastic_deform,debug)
    dataset.save(dst_location)
    return dataset

def load_dataset(dataset_name, use_elastic_deform, no_transformations):
    dst_location = get_path_data1(dataset_name, use_elastic_deform, no_transformations)
    return pkl.load(open(dst_location+"/dataset_test","rb"))
    
def train_model1(dataset_path,experiment_name):
    train_loop.train_loop(experiment_name,1,dataset_path)

def test_model1(dataset,experiment_name):
    train_loop.test_model( experiment_name, dataset, both_models=False)

def test_model2(dataset,experiment_name):
    train_loop.test_model( experiment_name, dataset, both_models=True)
    
    
def create_second_dataset(first_dataset_path,experiment_name):
    train_loop.create_second_dataset(first_dataset_path,experiment_name)
    
def train_model2(experiment_name):
    dataset_path = "/home/eduardo/Results/"+experiment_name+"/second_dataset/"
    train_loop.train_loop(experiment_name,2,dataset_path)

    
#dataset = pkl.load(open("/home/eduardo/dataset_elastic_trans_True/dataset_test","rb"))
#test_model2(dataset,"elastic_deforms_True")