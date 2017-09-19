# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:27:19 2017

@author: eduardo
"""
import sys
sys.path.append("/data/josecp/code/data/")
import cnn_models as models
import Data_INbreast
import utils as ut
import tensorflow as tf
import numpy as np
import glob

# TODO: ALEXNET WITH ROTATED CONVOLUTIONAL FILTERS
# TODO: PATCHES INBREAST
# TODO: CORRECT ACCURACY, AUC AND LOSS


def data_augment(batch):
    le = batch.shape[0]
    rots = np.random.randint(4,size = le)
    mirrs = np.random.randint(2,size = le)
    for i in range(le):
        batch[i] = np.rot90(batch[i],k=rots[i])
        if mirrs[i]:
            batch[i] = np.fliplr(batch[i])
    return batch
    
    
def test(experiment_name):
    
    epochs = 50
    iterations_per_epoch = 100
    learning_rate = 1e-4
    batch_size = 16
    
    tf.reset_default_graph()
    alex = models.alexNet(2)
    
    experiment_name = experiment_name
    results = np.zeros((epochs,6))
    
    paths = glob.glob("/var/tmp/inbreast*")
    if len(paths)==1:
        print("Loaded dataset")
        ddsa = Data_INbreast.load(paths[0]+"/DB_instance.pkl")
    else:
        print("Creating new dataset")
        ddsa = Data_INbreast.INbreast(batch_shape = [batch_size,224,224,1])
        ddsa.define_labels_based_on_birads()
        ddsa.split_train_test()
        ddsa.preprocess_and_save_imgs()
        ddsa.save_instance()
    
    tx,ty = ddsa.load_all_test()
    timer = ut.timer()
    
    print("Starting training: "+experiment_name+"\n epochs: {}; iterations_per_epoch: {}; learning_rate: {}; batch_size: {}".format(
                                                        epochs,iterations_per_epoch,learning_rate,batch_size))
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        #bar = ut.progress_bar(epochs)
        for i in range(epochs):
            print("")
            print("Epoch {}:".format(i))
            bar = ut.progress_bar(iterations_per_epoch)
            #bar.tick(i)
            preds_s = np.zeros(batch_size*iterations_per_epoch)
            y_s = np.zeros(batch_size*iterations_per_epoch)
            
            for j in range(iterations_per_epoch):
    
                batchx,batchy = ddsa.load_random_batch_train()
                batchx = data_augment(batchx)
                _,preds = alex.train(sess,batchx,batchy,learning_rate)
                preds_s[j*16:j*16+16] = preds[:,1]
                y_s[j*16:j*16+16] = batchy
                bar.tick()            
                
            metrics= ut.ut_all(preds[:,1],batchy)
            tr_loss= metrics[0]
            tr_acc =  metrics[1]
            tr_auc =  metrics[2]

            
            preds = alex.test(sess,tx)
    
            te_loss,te_acc,te_auc = 0,0,0        
            metrics = ut.ut_all(preds[:,1],ty)
            te_loss += metrics[0]
            te_acc +=  metrics[1]
            te_auc +=  metrics[2]
            
            results[i] = np.array([tr_loss,te_loss,tr_acc,te_acc,tr_auc,te_auc])
                
            print("\tLosses:\t\t{:.2f} -> {:.2f}".format(tr_loss,te_loss))
            print("\tAccuracies:\t{:.2f} -> {:.2f}".format(tr_acc,te_acc))
            print("\tROC - AUCs:\t{:.2f} -> {:.2f}".format(tr_auc,te_auc))
        
        path = "/var/tmp/"+experiment_name
        np.save(path,results)
        print("Results saved in: "+path)
        print("Finished: {} s".format(timer.tick()))
    

def main():
    for i in range(0,10):
        test("test_"+str(i))
        
if __name__ == "__main__":
    main()
        