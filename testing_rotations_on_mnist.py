#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 17:48:46 2017

@author: edu
"""
import cnns.cnn_models as models
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import cnns.cnn_lib as cnn_lib
import funcs.utils as ut

n_filters=4
def mnist_baseline():

    model = models.Model([28,28])
    model.add_easy_layer(ltype="conv",filters_shape = [5,5],n_filters = 4*n_filters, padding="SAME",stride=1,activation="relu")
    model.add_easy_layer(ltype="conv",filters_shape = [3,3],n_filters = 8*n_filters, padding="SAME",stride=1,activation="relu")
    model.add_easy_layer(ltype="max_pool",k=2, stride=2)
    
    model.add_easy_layer(ltype="conv",filters_shape = [3,3],n_filters = 16*n_filters, padding="SAME",stride=1,activation="relu")    
    model.add_easy_layer(ltype="max_pool",k=2, stride=2)
    
    model.add_easy_layer(ltype="flatten")
    model.add_easy_layer(ltype="dense", n_filters = 64*n_filters)
    
    model.add_easy_layer(ltype="dense",n_filters = 256)
    model.add_easy_layer(ltype="out", n_filters = 10)
    
    model._compile()
    print("Normal model")
    return model
    
    
def mnist_rotat_loss():
    reg_param_vars = 0.1
    
    model = models.Model([28,28])
    model.add_easy_layer(ltype="conv",filters_shape = [5,5],n_filters = 4*n_filters, padding="SAME",stride=1,activation="relu")
    model.add_easy_layer(ltype="conv",filters_shape = [3,3],n_filters = 8*n_filters, padding="SAME",stride=1,activation="relu")
    model.add_easy_layer(ltype="max_pool",k=2, stride=2)
    
    model.add_easy_layer(ltype="conv",filters_shape = [3,3],n_filters = 16*n_filters, padding="SAME",stride=1,activation="relu")    
    model.add_easy_layer(ltype="max_pool",k=2, stride=2)
    
    model.add_easy_layer(ltype="flatten")
    model.add_easy_layer(ltype="dense", n_filters = 64*n_filters)
    
    model.add_easy_layer(ltype="dense",n_filters = 256)
    model.add_easy_layer(ltype="out", n_filters = 10)
    
    
    
    print("Added variational loss")
    reshaped_layer = tf.reshape(model.layers[6].out,shape=[-1,4,1024])
    _,add_loss = tf.nn.moments(reshaped_layer,axes = [1])
    add_loss = tf.reduce_mean(add_loss)
    
    model._compile(add_loss = add_loss*reg_param_vars)
    return model

def mnist_rotat_corrected():
    
    model = models.Model([28,28])
    model.add_easy_layer(ltype="conv_rotat",filters_shape = [5,5],n_filters = 4*n_filters, padding="SAME",stride=1,activation="relu")
    model.add_easy_layer(ltype="conv_rotat_corrected",filters_shape = [3,3],n_filters = 8*n_filters, padding="SAME",stride=1,activation="relu")
    model.add_easy_layer(ltype="max_pool",k=2, stride=2)
    
    model.add_easy_layer(ltype="conv_rotat_corrected",filters_shape = [3,3],n_filters = 16*n_filters, padding="SAME",stride=1,activation="relu")    
    model.add_easy_layer(ltype="max_pool",k=2, stride=2)
       
    model.add_easy_layer(ltype="conv_rotat_corrected",filters_shape = [7,7],n_filters = 64*n_filters, padding="VALID",stride=1,activation="linear")
    model.add_easy_layer(ltype="flatten")
    
    model.add_easy_layer(ltype="dense",n_filters = 256)
    model.add_easy_layer(ltype="out", n_filters = 10)
    
    print("Architecture rotat")
    model._compile()
    
    return model

def mnist_rotat_both():
    
    reg_param_vars = 0.1
    
    model = models.Model([28,28])
    model.add_easy_layer(ltype="conv_rotat",filters_shape = [5,5],n_filters = 4*n_filters, padding="SAME",stride=1,activation="relu")
    model.add_easy_layer(ltype="conv_rotat_corrected",filters_shape = [3,3],n_filters = 8*n_filters, padding="SAME",stride=1,activation="relu")
    model.add_easy_layer(ltype="max_pool",k=2, stride=2)
    
    model.add_easy_layer(ltype="conv_rotat_corrected",filters_shape = [3,3],n_filters = 16*n_filters, padding="SAME",stride=1,activation="relu")    
    model.add_easy_layer(ltype="max_pool",k=2, stride=2)
    
    model.add_easy_layer(ltype="conv_rotat_corrected",filters_shape = [7,7],n_filters = 64*n_filters, padding="VALID",stride=1,activation="linear")
    model.add_easy_layer(ltype="flatten")
    
    model.add_easy_layer(ltype="dense",n_filters = 256)
    model.add_easy_layer(ltype="out", n_filters = 10)
    
    print("Architecture and variational loss")
    reshaped_layer = tf.reshape(model.layers[6].out,shape=[-1,4,1024])
    _,add_loss = tf.nn.moments(reshaped_layer,axes = [1])
    add_loss = tf.reduce_mean(add_loss)
    
    model._compile(add_loss = add_loss*reg_param_vars)
    return model


def rotate_images(batchx):
    num = batchx.shape[0]
    rnd = np.random.randint(0,4,num)
    for i in range(num):
        batchx[i] = np.rot90(batchx[i],rnd[i],axes=[0,1])
    return batchx
    
tf.reset_default_graph()
Iterations = int(10e4)
Test = 500

model = mnist_rotat_corrected()

save_results_num = int(Iterations/Test)
Losses = np.zeros(save_results_num)
Accuracies = np.zeros(save_results_num)

Test_Losses = np.zeros(save_results_num)
Test_Accuracies = np.zeros(save_results_num)

first = True

loss_list=[]
accs_list=[]

Tloss_list=[]
Taccs_list=[]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    counter = 0
    for iteration in range(Iterations):

        batchx,batchy = mnist.train.next_batch(128)
        batchx = np.reshape(batchx,[-1,28,28,1])
        batchx = rotate_images(batchx)
        batchy = np.argmax(batchy,axis=1)
        loss,acts = model.train(sess,batchx,batchy,1e-3)
        
        loss_list.append(loss)
        accs_list.append(np.mean(np.argmax(acts,axis=1)==batchy))
        
        
        
        if iteration%Test == 0:
            print(iteration/Iterations)
            if first:
                first=False
                continue
            
            xx = mnist.validation.images[0:mnist.validation.num_examples]
            xx = np.reshape(xx,[-1,28,28,1])
            xx = rotate_images(xx)
            yy = mnist.validation.labels[0:mnist.validation.num_examples]
            yy = np.argmax(yy,axis=1)
            
            for i in range(0,mnist.validation.num_examples,1000):
                loss,acts = model.test(sess,xx[i:i+1000],yy[i:i+1000])
                Tloss_list.append(loss)
                Taccs_list.append(np.mean(np.argmax(acts,axis=1)==yy[i:i+1000]))
            
            Losses[counter] = np.mean(loss_list)
            Accuracies[counter] = np.mean(accs_list)
            
            Test_Losses[counter] = np.mean(Tloss_list)
            Test_Accuracies[counter] = np.mean(Taccs_list)
            
            loss_list=[]
            accs_list=[]

            Tloss_list=[]
            Taccs_list=[]
            counter+=1
"""            

img = np.reshape(mnist.train.images[0],[28,28])
model = mnist_rotat_corrected()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

res = model.test_layer(sess,img[np.newaxis,:,:,np.newaxis],1)[0]

f,ax = plt.subplots(4,2)
ax[0,0].imshow(res[0,:,:,0])
ax[0,1].imshow(res[0,:,:,1])
ax[1,0].imshow(res[0,:,:,2])
ax[1,1].imshow(res[0,:,:,3])
ax[2,0].imshow(res[0,:,:,4])
ax[2,1].imshow(res[0,:,:,5])
ax[3,0].imshow(res[0,:,:,6])
ax[3,1].imshow(res[0,:,:,7])

img = np.rot90(img)
res = model.test_layer(sess,img[np.newaxis,:,:,np.newaxis],1)[0]

f,ax = plt.subplots(4,2)
ax[0,0].imshow(np.rot90(res[0,:,:,0],k=3))
ax[0,1].imshow(np.rot90(res[0,:,:,1],k=3))
ax[1,0].imshow(np.rot90(res[0,:,:,2],k=3))
ax[1,1].imshow(np.rot90(res[0,:,:,3],k=3))
ax[2,0].imshow(np.rot90(res[0,:,:,4],k=3))
ax[2,1].imshow(np.rot90(res[0,:,:,5],k=3))
ax[3,0].imshow(np.rot90(res[0,:,:,6],k=3))
ax[3,1].imshow(np.rot90(res[0,:,:,7],k=3))
"""

"""
plt.imshow(img)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


W = models.Param([3,3,4,1],"W321",models.initializer("normal",1)).value




Wr = sess.run(W)
f,ax = plt.subplots(1,4)
ax[0].imshow(Wr[:,:,0,0])
ax[1].imshow(Wr[:,:,1,0])
ax[2].imshow(Wr[:,:,2,0])
ax[3].imshow(Wr[:,:,3,0])

W = cnn_lib.shift(W)
Wr = sess.run(W)
f,ax = plt.subplots(1,4)
ax[0].imshow(Wr[:,:,0,0])
ax[1].imshow(Wr[:,:,1,0])
ax[2].imshow(Wr[:,:,2,0])
ax[3].imshow(Wr[:,:,3,0])



res = model.test_layer(sess,img[np.newaxis,:,:,np.newaxis],-1)[0]

f,ax = plt.subplots(4,2)
ax[0,0].imshow(res[0,:,:,0])
ax[0,1].imshow(res[0,:,:,1])
ax[1,0].imshow(res[0,:,:,2])
ax[1,1].imshow(res[0,:,:,3])
ax[2,0].imshow(res[0,:,:,4])
ax[2,1].imshow(res[0,:,:,5])
ax[3,0].imshow(res[0,:,:,6])
ax[3,1].imshow(res[0,:,:,7])

img = np.rot90(img)
res = model.test_layer(sess,img[np.newaxis,:,:,np.newaxis],-1)[0]

f,ax = plt.subplots(4,2)
ax[0,0].imshow(np.rot90(res[0,:,:,0],k=3))
ax[0,1].imshow(np.rot90(res[0,:,:,1],k=3))
ax[1,0].imshow(np.rot90(res[0,:,:,2],k=3))
ax[1,1].imshow(np.rot90(res[0,:,:,3],k=3))
ax[2,0].imshow(np.rot90(res[0,:,:,4],k=3))
ax[2,1].imshow(np.rot90(res[0,:,:,5],k=3))
ax[3,0].imshow(np.rot90(res[0,:,:,6],k=3))
ax[3,1].imshow(np.rot90(res[0,:,:,7],k=3))

"""
