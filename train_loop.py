# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 17:20:10 2017

@author: eduardo
"""

import funcs.patch_loader as pl
import tensorflow as tf
import funcs.utils as ut
import cnns.cnn_models as models
import numpy as np
import os
import funcs.image_processing as iproc
from matplotlib import pyplot as plt
import pickle as pkl

def print_s(string,file):
    print(string)
    print(string,file=file)
    

def train_loop(experiment_name, model_number, dataset_path):
    """
    DEFINE PARAMETERS
    """
    #experiment_name = "test_experiment"
    #model_number = 1 # DEFINES IF THIS IS THE FIRST OR SECOND MODEL IN THE CASCADE
    model = models.detector36(False, "model"+str(model_number), False) # MODEL ARCHITECTURE
    
    
    # LEARNING PARAMETERS
    learning_rate = 0.0001
    
    training_iterations = int(5e3)  # TRAINING ITERATIONS
    one_epoch_every = 1e2           # ITERATIONS PER EPOCH 
    number_of_epochs = int(training_iterations/one_epoch_every)+1
    
    """
    PATH:
        RESULTS
    """
    if model_number == 1:
        results_path = "/home/eduardo/Results/"+experiment_name
        os.mkdir(results_path)
    
    
    log_file = open(results_path+"/log.txt","w") # LOG FILE
    metrics_save_path = results_path+"/metrics.npy"
    
    save_weights_path = results_path+"/model"+str(model_number)
    os.mkdir(save_weights_path)
    load_weights_path = None
    if model_number == 2:
        load_weights_path = results_path+"/model1"
    
    """
        DATASET
    """
    #dataset_path = "/home/eduardo/dataset_name"
    if model_number == 2:
        dataset_path += "second"
        
    
    
    evaluation_metric_index = 1 #DEFINES THE POSITION OF THE METRIC USED FOR COMPARISSON
    evaluation_metric_objective = 1 #1 in case of Maximization and -1 in case of Minimization (ONLY POSITIVE VALUES)
    number_of_evaluation_metrics = 4
    metrics_array = np.zeros((number_of_epochs,number_of_evaluation_metrics*2+1))
    max_evaluation_metric = -1e9
    
    """
    CONFIGURE PATCH LOADER
    """
    pl.define(dataset_path, ["negative","positive"], 128, (36,36,1))
    pl.load_random_batch("train")
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    acts_list, loss_list, y_list = [],[],[]
    
    
    
    if load_weights_path:
        model.load(sess,load_weights_path)
    
    epoch_counter = 0
    
    for iteration in range(training_iterations):
        
        # GET LOADED BATCH
        batchx,batchy = pl.get_prepared_batch()    
        # START LOADING NEXT BATCH
        pl.load_random_batch("train")
        # TRAIN THE MODEL ON THE CURRENT BATCH    
        loss,acts = model.train(sess,batchx,batchy,learning_rate)
        
        # APPEND ITERATION RESULTS IN THE LISTS
        acts_list.append(acts)
        loss_list.append(loss)
        y_list.append(batchy)
        
        if (iteration)%one_epoch_every == 0:
    
            split_counter = pl.iterative_load_full_split("validation",0)
            val_acts_list, val_loss_list, val_y_list = [],[],[]
            
            while True:
                val_batchx,val_batchy = pl.get_prepared_batch()

                split_counter = pl.iterative_load_full_split("validation",split_counter)
                    
                val_loss,val_acts = model.test(sess,val_batchx,val_batchy)
                
                val_acts_list.append(val_acts)
                val_loss_list.append(val_loss)
                val_y_list.append(val_batchy)
                
                if split_counter == 0:
                    
                    val_batchx,val_batchy = pl.get_prepared_batch()
                    
                    val_loss,val_acts = model.test(sess,val_batchx,val_batchy)
                
                    val_acts_list.append(val_acts)
                    val_loss_list.append(val_loss)
                    val_y_list.append(val_batchy)
                    
                    break
        
            pl.load_random_batch("train")
              
            # COMPUTE METRICS FOR THE TRAINING SET
            acts_list = ut.concatenate_arrays(acts_list)[:,1]
            loss_list = np.array(loss_list)
            y_list = ut.concatenate_arrays(y_list)
            metrics = ut.ut_metrics(acts_list, y_list,loss_list)
            acts_list, loss_list, y_list = [],[],[]
            
            # COMPUTE METRICS FOR THE VALIDATION SET
            val_acts_list = ut.concatenate_arrays(val_acts_list)[:,1]
            val_loss_list = np.array(val_loss_list)
            val_y_list = ut.concatenate_arrays(val_y_list)
            val_metrics = ut.ut_metrics(val_acts_list, val_y_list,val_loss_list)
            
            # PRINT RESULTS
            result_string = "Iter: "+str(iteration)
            result_string += " | Train: "
            for metric in metrics:
                result_string += (metric[0] + ": " + str(metric[1]) + ", ")

            # PRINT RESULTS
            result_string += " | Validation: "
            for metric in val_metrics:
                result_string += (metric[0] + ": " + str(metric[1]) + ", ")
    
            print_s(result_string,log_file)
            
            # SAVE MODEL IF IT IS THE BEST
            curr_evaluation_metric = evaluation_metric_objective * val_metrics[evaluation_metric_index][1]
            if curr_evaluation_metric>max_evaluation_metric:
                max_evaluation_metric = curr_evaluation_metric
                model.save(sess,save_weights_path)
            
            metrics_array[epoch_counter,0] = epoch_counter
            i=1        
            for metric in metrics+val_metrics:
                metrics_array[epoch_counter,i] = metric[1]
                i+=1
                
            epoch_counter+=1
    
    
    log_file.close()
    np.save(metrics_save_path,metrics_array[0:epoch_counter,:])
    sess.close()
    

# TODO WITH MODEL NUMBER: IDEA -> Use recursion
def test_model(model_num, experiment_name, dataset_first):
    
    tf.reset_default_graph()
    sess = tf.Session()
    
    results_path = "/home/eduardo/Results/"+experiment_name
    os.mkdir(results_path+"/heat_maps"+str(model_num))
    load_weights_path = results_path+"/model1"    
    model,model_full = load_model(sess,model_num,load_weights_path)
    iDs = dataset_first.files_names.keys()
    
    all_suspicions = dict()
    bar = ut.progress_bar(len(iDs))
    for iD in iDs: # TODO
        bar.tick()
        all_suspicions[iD] = list()
        image = np.load(dataset_first.files_names[iD])
        htmap = model_full.test(sess,image)
        htmap = iproc.filter_img(htmap)
        htmap = iproc.improved_non_maxima_supression(htmap)
        np.save(results_path+"/heatmaps/"+os.path.basename(dataset_first.files_names[iD]),htmap)
        dets = iproc.detections(htmap,10)
        
        masks = []
        masks_files = dataset_first.masks[iD]
        for file in masks_files:
            mask = np.load(file)
            masks.append(mask)
        
        masks_hit = np.zeros(len(masks))
        for det in dets:
            correct_mask = inside_masks(det,masks,masks_hit)
            if correct_mask!=-1:
                if not masks_hit[correct_mask]:
                    all_suspicions[iD].append([*det,"TP"])
                    masks_hit[correct_mask]=1
            else:
                all_suspicions[iD].append([*det,"FP"])
        
        for i in range(len(masks_hit)):
            if masks_hit[i] == 0:
                all_suspicions[iD].append([-1,-1,"FN"])
    
    pkl.dump(all_suspicions,open(results_path+"/all_suspicions","wb"))
    compute_score(all_suspicions,results_path) 
    
    sess.close()

    return all_suspicions
    
def compute_score(all_suspicions,results_path=""):
    num_of_masses = 0    
    num_of_images = 0
    for image in all_suspicions.keys():
        num_of_images+=1
        for susp in all_suspicions[image]:
            if susp[2] in ["TP","FN"]:
                num_of_masses+=1
    
    print("Working with:",num_of_masses," masks")
    print("Working with:",num_of_masses," images")    
    
    tp = np.zeros(num_of_masses)
    fp = list()
    counter=0
    for image in all_suspicions.keys():
        for susp in all_suspicions[image]:
            if susp[2] == "TP":
                tp[counter] = susp[1]
                counter+=1
            elif susp[2] == "FP":
                fp.append(susp[1])
    
            
    finalTP = np.stack((tp,np.ones(tp.shape)),axis=1)
    fp = np.array(fp)    
    finalFP = np.stack((fp,np.zeros(fp.shape)),axis=1)
    final = np.concatenate((finalTP,finalFP),axis = 0)
    indexes = np.argsort(final[:,0])
    final[:] = final[indexes[::-1]]
    tpr_vec = np.cumsum(final[:,1])/num_of_masses
    fpi_vec = np.cumsum(final[:,1]==0)/num_of_images
    
    plt.scatter(fpi_vec,tpr_vec,s=0.1)
    pkl.dump([fpi_vec,tpr_vec],open(results_path+"/fpi_tpr_vecs","wb"))
    
    plt.savefig(results_path+"/Free_Roc_Curve")
    
    return final,fpi_vec,tpr_vec
        
def inside_masks(det,masks,masks_hit):
    for i in range(masks_hit.shape[0]):
        center = det[0]
        mask = masks[i]
        #plt.imshow(mask)
        #plt.show()
        #print(center)
        if mask[center[0],center[1]]:
            return i
    
    return -1
    
def load_model(sess,model_num,load_weights_path):
    model = models.detector36(False, "model"+str(model_num), False)
    model.load(sess,load_weights_path)    
    model_full = models.detector36(True, "model"+str(model_num), True)    
    return model,model_full
    

    