# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:20:23 2017

@author: eduardo
"""
import cnn_models as models
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import utils as ut


def data_augmentation_func(x):
    le = x.shape[0]
    rots = np.random.randint(4,size = le)
    #mirrs = np.random.randint(2,size = le)
    for i in range(le):
        x[i] = np.rot90(x[i],k=rots[i])
        #if mirrs[i]:
        #    x[i] = np.fliplr(x[i])
    return x
    
def data_expansion_func(x,y):
    vals = list()
    
    for i in range(4):
        vals.append(np.rot90(x,k=i,axes=(1,2)))

    tx = np.concatenate(vals)         
    ty = np.concatenate([y,y,y,y])
    
    return tx,ty


def data_var(x,y):
    batch2 = np.zeros((x.shape[0]*4,*x.shape[1::]))
    y2 = np.zeros((x.shape[0]*4))
    for i in range(x.shape[0]):
        for innK in range(4):
            batch2[i*4+innK] = np.rot90(x[i], k=innK, axes = (0,1))
            y2[i*4+innK] = y[i]
    return batch2,y2

def make_shuffle():
    global trainx,trainy
    
    num_examples = trainx.shape[0]
    
    perm0 = np.arange(num_examples)
    np.random.shuffle(perm0)
    trainx = trainx[perm0]
    trainy = trainy[perm0]
    
    
def next_batch(batch_size):
    global _t_index
    
    num_examples = trainx.shape[0]
    
    if _t_index + batch_size > num_examples:
        
        end = num_examples - _t_index
        batch[0:end,:,:,0] = trainx[_t_index:num_examples]
        batchy[0:end] = trainy[_t_index:num_examples]
        make_shuffle()
        
        end2 = batch_size-end
        _t_index = 0
        batch[end:end+end2,:,:,0] = trainx[_t_index:end2]
        batchy[end:end+end2] = trainy[_t_index:end2]
        _t_index = end2
        
        assert (end+end2) == batch_size
        
        return batch,batchy
    
    else:
        
        batch[0:batch_size,:,:,0] = trainx[_t_index:_t_index+batch_size]
        batchy[:] = trainy[_t_index:_t_index+batch_size]
        _t_index += batch_size
        return batch,batchy
        
        

#def get_batch(images)
    
def test(model,path_for_results, data_augmentation, data_expansion, special):
    
    global trainx,trainy
    epochs = 200
    iterations = 20

    results = np.zeros((epochs,3))

    trainx = np.reshape(mnist.train.images[0:150],[-1,28,28])
    trainy = mnist.train.labels[0:150]
    
    valx = np.reshape(mnist.validation.images,[-1,28,28])
    valy = mnist.validation.labels
            
    if data_expansion:
        valx,valy = data_expansion_func(valx,valy)

    valx = valx[:,:,:,np.newaxis]
    
    
    with tf.Session() as sess:
    
        sess.run(tf.global_variables_initializer())
        
        print("starting: " + path_for_results)
        bar = ut.progress_bar(epochs)
        for epoch in range(epochs):
            bar.tick()
            e_loss = 0
            e_acc  = 0

            for iteration in range(iterations):
                batchx,batchy = next_batch(batch_size)                
                #batchx,batchy = mnist.train.next_batch(128)
                #batchx = np.reshape(batchx,[-1,28,28])
                
                if special:
                    bx,by = data_var(batchx,batchy)
                
                elif data_augmentation:
                    bx = data_augmentation_func(batchx)
                    by = batchy
                #batchx = batchx[:,:,:,np.newaxis]
                
                loss,preds = model.train(sess,bx,by,0.2e-3)
                e_loss += loss/iterations
                e_acc += (np.argmax(preds , axis = 1) == by).sum()/128/iterations
            
            
            
            #result = sess.run(model.layers[6].out,feed_dict={model.inp:bx})
            total_preds = []
            for local_i in range(0,valx.shape[0],200):
                total_preds.append(model.test(sess,valx[local_i:local_i+200]))
            preds = np.concatenate(total_preds)
            val_acc = ut.acc_multiclass(preds,valy,np.ones(10))
            
            results[epoch,0] = e_loss        
            results[epoch,1] = e_acc
            results[epoch,2] = val_acc
            #print(epoch, e_loss, e_acc, val_acc)
            
        np.save(path_for_results,results)
        print(np.max(results,axis=0))
            


mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

_t_index = 0
batch_size = 32
batch = np.zeros((batch_size,28,28,1))
batchy = np.zeros((batch_size))

tf.reset_default_graph()
model = models.mnist_soft_inv_arch(reg_param=1)
test(model,"/var/tmp/mnist_s_soft",True,True,True)

_t_index = 0
batch_size = 128
batch = np.zeros((batch_size,28,28,1))
batchy = np.zeros((batch_size))
tf.reset_default_graph()
model = models.mnist_arch()
test(model,"/var/tmp/mnist_s_normal",True,True,False)

_t_index = 0
batch_size = 128
batch = np.zeros((batch_size,28,28,1))
batchy = np.zeros((batch_size))
tf.reset_default_graph()
model = models.mnist_arch(True,v1=True)
test(model,"/var/tmp/mnist_s_rotat",True,True,False)

_t_index = 0
batch_size = 128
batch = np.zeros((batch_size,28,28,1))
batchy = np.zeros((batch_size))
tf.reset_default_graph()
model = models.mnist_arch(True,v1=False)
test(model,"/var/tmp/mnist_s_rotat2",True,True,False)