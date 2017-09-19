# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 17:20:10 2017

@author: eduardo
"""

import patch_loader as pl
import tensorflow as tf
import funcs.utils as ut


"""
DEFINE PARAMETERS
"""
experiment_name = None
experiment_description = None
model = None
learning_rate = 0.0001
batch_size = 128
shape = (36,36,1)
training_iterations = 1e6
one_epoch_every = 1e4
load_weights_path = False # LEAVE FALSE TO START FROM SCRATCH
save_weights_path = False # LEAVE FALSE SO THE MODEL IS NOT SAVED

results_path = None
results_file = open("results_path","w")

dataset_path = None
dataset_classes = ["positive","negative"]
evaluation_metric_index = 0 #DEFINES THE POSITION OF THE METRIC USED FOR COMPARISSON
evaluation_metric_objective = 1 #1 in case of Maximization and -1 in case of Minimization


"""
CONFIGURE PATCH LOADER
"""
pl.define_classes(dataset_classes)
pl.define_size_shape(batch_size,shape)
pl.define_initial_path(dataset_path)
pl.load_random_batch("train")

sess = tf.Session()
sess.run(tf.global_variables_initializer())
acts_list, loss_list, y_list = [],[],[]

for iteration in range(training_iterations):
    
    # GET LOADED BATCH
    batchx,batchy = pl.get_prepared_batch()    
    # START LOADING NEXT BATCH
    pl.load_random_batch("train")
    # TRAIN THE MODEL ON THE CURRENT BATCH    
    acts,loss = model.train(sess,batchx,batchy,learning_rate)
    
    # APPEND ITERATION RESULTS IN THE LISTS
    acts_list.append(acts)
    loss_list.append(loss)
    y_list.append(batchy)
    
    if iteration%one_epoch_every == 0:

        pl.load_specific_batch("validation",0)
        val_acts_list, val_loss_list, val_y_list = [],[],[]
        
        for val_iter in range(pl.number_of_batches("validation")):

            val_batchx,val_batchy = pl.get_prepared_batch()
            pl.load_specific_batch("validation",val_iter+1)
            val_acts,val_loss = model.test(sess,batchx,batchx)
            
            val_acts_list.append(acts)
            val_loss_list.append(loss)
            val_y_list.append(val_batchy)
    
        pl.load_random_batch("train")
        acts_list, loss_list, y_list = [],[],[]
          
        # COMPUTE METRICS FOR THE TRAINING SET
        acts_list = ut.concatenate_arrays(acts_list)
        loss_list = ut.concatenate_arrays(loss_list)
        y_list = ut.concatenate_arrays(y_list)
        metrics = ut.standard_metrics(acts_list, loss_list, y_list)
    
        # COMPUTE METRICS FOR THE VALIDATION SET
        val_acts_list = ut.concatenate_arrays(val_acts_list)
        val_loss_list = ut.concatenate_arrays(val_loss_list)
        val_y_list = ut.concatenate_arrays(val_y_list)
        val_metrics = ut.standard_metrics(val_acts_list, val_loss_list, val_y_list)
            
        print("\nI",str(iteration),end=" ")
        # PRINT RESULTS
        print("Train -> ",end = " ")
        for metric in metrics:
            print(metric[0],metric[1],end=" ")
            
        # PRINT RESULTS
        print("VALIDATION -> ",end = " ")
        for metric in val_metrics:
            print(metric[0],metric[1],end=" ")

        
    
