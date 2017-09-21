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

def print_s(string,file):
    print(string)
    print(string,file=file)
    
    
"""
DEFINE PARAMETERS
"""
experiment_name = "test_experiment"
experiment_description = "This is a simple test to check if everything is working as intended"
model = models.detector36()
learning_rate = 0.0001
batch_size = 128
shape = (36,36,1)
training_iterations = int(1e4)
one_epoch_every = 1e2
load_weights_path = False # LEAVE FALSE TO START FROM SCRATCH
save_weights_path = False # LEAVE FALSE SO THE MODEL IS NOT SAVED

results_path = "test_experiment_results"
results_file = open(results_path,"w")
model_restore_path = False
model_save_path = "save_model_test"

dataset_path = "/home/eduardo/tmp_testdataset2ndpart"
dataset_classes = ["positive","negative"]
evaluation_metric_index = 0 #DEFINES THE POSITION OF THE METRIC USED FOR COMPARISSON
evaluation_metric_objective = 1 #1 in case of Maximization and -1 in case of Minimization
max_evaluation_metric = -1e9

"""
CONFIGURE PATCH LOADER
"""
pl.define_initial_path(dataset_path)
pl.define_classes(dataset_classes)
pl.define_size_shape(batch_size,shape)
pl.load_random_batch("train")

sess = tf.Session()
sess.run(tf.global_variables_initializer())
acts_list, loss_list, y_list = [],[],[]



if model_restore_path:
    model.load(sess,model_restore_path)


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
    
    if (iteration+1)%one_epoch_every == 0:

        pl.load_specific_batch("validation",0)
        val_acts_list, val_loss_list, val_y_list = [],[],[]
        
        for val_iter in range(pl.number_of_batches("validation")):

            val_batchx,val_batchy = pl.get_prepared_batch()
            pl.load_specific_batch("validation",val_iter+1)
            val_loss,val_acts = model.test(sess,val_batchx,val_batchy)
            
            val_acts_list.append(acts)
            val_loss_list.append(loss)
            val_y_list.append(val_batchy)
    
        pl.load_random_batch("train")
          
        # COMPUTE METRICS FOR THE TRAINING SET
        acts_list = ut.concatenate_arrays(acts_list)[:,1]
        loss_list = np.array(loss_list)
        y_list = ut.concatenate_arrays(y_list)
        metrics = ut.ut_metrics(acts_list, y_list)
        acts_list, loss_list, y_list = [],[],[]
        
        # COMPUTE METRICS FOR THE VALIDATION SET
        val_acts_list = ut.concatenate_arrays(val_acts_list)[:,1]
        val_loss_list = np.array(val_loss_list)
        val_y_list = ut.concatenate_arrays(val_y_list)
        val_metrics = ut.ut_metrics(val_acts_list, val_y_list)
        
        # PRINT RESULTS
        result_string = "Iter: "+str(iteration)
        result_string += " | Train: "
        for metric in metrics:
            result_string += (metric[0] + ": " + str(metric[1]) + ", ")
            
        # PRINT RESULTS
        result_string += " | Validation: "
        for metric in val_metrics:
            result_string += (metric[0] + ": " + str(metric[1]) + ", ")

        print_s(result_string,results_file)
        
        # SAVE MODEL IF IT IS THE BEST
        curr_evaluation_metric = evaluation_metric_objective * val_metrics[evaluation_metric_index][1]
        if curr_evaluation_metric>max_evaluation_metric:
            max_evaluation_metric = curr_evaluation_metric
            model.save(sess,model_save_path)


results_file.close()
sess.close()