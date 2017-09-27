# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 17:43:39 2017

@author: eduardo
"""

import data.cbis_standart_format as cbis_standart_format
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
    cbis_standart_format.make_CBIS_standart_format(1/inv_scale,clahe,src_location,folder_images,folder_masks,dst_location,debug)
    
    
def get_path_data1(dataset,use_elastic_deform,no_transformations):
    use_elastic_deform_text = ""
    if use_elastic_deform:
        use_elastic_deform_text += "_deform_elas"
    return "/home/eduardo/data_"+dataset+"_"+str(no_transformations)+use_elastic_deform_text+"/"
    
def create_first_dataset(src_location, dataset_name, use_elastic_deform, no_transformations):
    
    dst_location = get_path_data1(dataset_name, use_elastic_deform, no_transformations)
    
    set_masses_size = {"train":491,"validation":100,"test":100}
    set_neg_size = {"train":0,"validation":0,"test":0}
    
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
    train_loop.test_model(1, experiment_name, dataset, sigma=0.8, num_dets=40)

def create_second_dataset(first_dataset_path,experiment_name):
    train_loop.create_second_dataset(first_dataset_path,experiment_name)
    
def train_model2():
    train_loop.train_loop(experiment_name,2,dataset_path)

def test_model2():
    #TODO IMPORTANT