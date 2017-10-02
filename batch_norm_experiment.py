# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 18:24:35 2017

@author: eduardo
"""
import configs_funcs as cfunc
import os
import pickle as pkl
import train_loop
import tensorflow as tf



dataset = "inbreast"
inv_scale = 24
clahe = True


first_dataset_name = dataset
use_elastic_deform = True
no_transformations = 40



pre_data_folder = cfunc.get_path_predata(dataset,inv_scale,clahe)

if not os.path.isdir(pre_data_folder):
    cfunc.create_standart_predata(dataset,inv_scale,clahe,debug=False)
    
data_folder = cfunc.get_path_data1(first_dataset_name, use_elastic_deform, no_transformations)

if not os.path.isdir(data_folder):
    dataset = cfunc.create_first_dataset(pre_data_folder, first_dataset_name , use_elastic_deform, no_transformations)
else:
    dataset = cfunc.load_dataset(first_dataset_name, use_elastic_deform, no_transformations)


dataset_path = cfunc.get_path_data1(first_dataset_name, use_elastic_deform, no_transformations)

results = dict()
for batch in [True, False]:
    for dropout in [True, False]:
        tf.reset_default_graph()
        experiment_name = "Batch_Dropout_exp_"+str(batch)+"_"+str(dropout)
        metrics = train_loop.train_loop(experiment_name,1,dataset_path,batch_norm=batch,dropout=dropout)
        max_pr = metrics[:,6].max()
        max_roc = metrics[:,7].max()
        max_acc = metrics[:,8].max()
        
        results[batch,dropout] = max_pr
        results[batch,dropout] = max_roc
        results[batch,dropout] = max_acc
        
pkl.dump(results,open("/home/eduardo/Results/Batch_Dropout_exp_results","wb"))
        