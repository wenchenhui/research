# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:21:22 2017

@author: Eduardo Meca Castro

Nones:

-> When reusing variables in functions get_conv_layer() and 
get_dense_layer() batch normalization is implemented with global statistics
rather than having a train and a test phase


"""
import tensorflow as tf
import numpy as np

""" 
Creates a convolutional layer to be used in a model
Params:     name        -> name for variable scope (string)
            inp         -> input to the layer (tensor)     
            shape       -> weights shape (list of ints)
            phase_train -> bool for batch norm (use None for no batch norm)
            reuse       -> wether the weights should be reused or not (bool)
"""

def get_conv_layer(name,inp,shape,phase_train):
    
    with tf.name_scope('convolution') as scope:
        # GET PARAMETERS AND PERFORM CONVOLUTION 
        W,b = get_params(shape, name)
        params = [W,b]
        layer = conv2d(inp,W,b,1,name=name)
        
        if phase_train!=None and phase_train!=False:
            # GET BATCH NORMALIZATION PARAMETERS
            beta,gamma = get_bn_params(shape[-1], name)
            batch_mean, batch_var = tf.nn.moments(layer, [0,1,2], name='global_stats_'+name)
            ema = tf.train.ExponentialMovingAverage(decay=0.9)
            
            # PERFORM BATCH NORMALIZATION
            layer = batch_norm(layer,beta,gamma,phase_train,ema,batch_mean,batch_var)
            
            # ASSIGN PARAMETERS
            params = params+[beta,gamma,batch_mean,batch_var,ema]
            
        layer = tf.nn.relu(layer)
    
    return layer,params
    
def get_rotat_conv_layer(name,inp,shape,phase_train):
    
    with tf.name_scope('convolution') as scope:
        # GET PARAMETERS AND PERFORM CONVOLUTION 
        W,b = get_params(shape, name)
        params = [W,b]

        layers = []
        for i in range(4):
            layers.appennd(conv2d(inp,W,b,1,name=name))
            W = tf.transpose(tf.reverse(W,axis=0),perm=[0,1])
        layer = tf.concat(layers,axis = 3)
            
        if phase_train!=None and phase_train!=False:
            # GET BATCH NORMALIZATION PARAMETERS
            beta,gamma = get_bn_params(shape[-1], name)
            batch_mean, batch_var = tf.nn.moments(layer, [0,1,2], name='global_stats_'+name)
            ema = tf.train.ExponentialMovingAverage(decay=0.9)
            
            # PERFORM BATCH NORMALIZATION
            layer = batch_norm(layer,beta,gamma,phase_train,ema,batch_mean,batch_var)
            
            # ASSIGN PARAMETERS
            params = params+[beta,gamma,batch_mean,batch_var,ema]
            
        layer = tf.nn.relu(layer)
    
    return layer,params
    
    
def get_conv_layer_test_only(inp,params):
    
    if len(params)==2:
        W,b = params
        bnorm = False
    elif len(params)==7:
        W,b,beta,gamma,batch_mean,batch_var,ema = params
        bnorm = True
    else:
        raise Exception("Wrong number of parameters")

    # CONVOLUTIONAL LAYER
    layer = conv2d(inp,W,b,1)
    
    # BATCH NORMALIZATION
    if bnorm:
        layer = tf.nn.batch_normalization(layer,ema.average(batch_mean),ema.average(batch_var),beta,gamma,1e-3)
    
    # RELU LAYER
    layer = tf.nn.relu(layer)
    
    return layer


"""
def get_conv_layer(name,inp,shape,phase_train,reuse = False):
    
    # GET PARAMETERS AND PERFORM CONVOLUTION 
    W,b = get_params(shape, name, reuse = reuse)
    layer = conv2d(inp,W,b,1,name=name)
    
    if phase_train!=None:
        # GET BATCH NORMALIZATION PARAMETERS
        beta,gamma = get_bn_params(shape[-1], name, reuse = reuse)
        batch_mean, batch_var = tf.nn.moments(layer, [0,1,2], name='global_stats_'+name)
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        
        # PERFORM BATCH NORMALIZATION
        if phase_train == "test_only":
            layer = tf.nn.batch_normalization(layer,ema.average(batch_mean),ema.average(batch_var),beta,gamma,1e-3)
        else:
            layer = batch_norm(layer,beta,gamma,phase_train,ema,batch_mean,batch_var)
    
""" 

def conv2d(x,W,b,stride,padding="VALID"):
    x = tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding=padding)
    x = tf.nn.bias_add(x,b)
    return x
    
def rotate(W):
    W = tf.transpose(W,perm=[1,0,2,3])
    W = tf.reverse(W,axis=[0])
    return W
""" 
Creates a max pooling layer to be used in a model
Params:     inp         -> input to the layer (tensor)     
            k           -> value used both for stride and kernel size (int)
"""
def maxpool2d(inp,k=2,stride=None):
    if stride==None:
        stride=k
    return tf.nn.max_pool(inp,ksize = [1,k,k,1],strides = [1,stride,stride,1], padding="SAME")


def full_image_max_pool(inp,k=2):
    layer = tf.nn.max_pool(inp,ksize=[1,k,k,1],strides = [1,k,k,1],padding="SAME")
    return strided(layer)
"""
Flattens a tensor. Used before the fully connected part of a network
"""
def flatten(x):
    shape = x.get_shape().as_list()
    #print("flattened shape:",shape)
    return tf.reshape(x,[-1,shape[1]*shape[2]*shape[3]])
    
""" 
Creates a convolutional layer to be used in a model
Params:     name        -> name for variable scope (string)
            inp         -> input to the layer (tensor)     
            shape       -> weights shape (list of ints)
            phase_train -> bool for batch norm (use None for no batch norm)
            reuse       -> wether the weights should be reused or not (bool)
"""
def get_dense_layer(name,inp,shape,phase_train,actv=True):
    
    # GET PARAMETERS AND PERFORM LINEAR CONV
    W,b = get_params(shape,name)
    params = [W,b]
    layer = dense(inp,W,b)
    
    
    if phase_train!=None and phase_train!=False:
        # GET BATCH NORMALIZATION PARAMETERS
        beta,gamma = get_bn_params(shape[-1],name)
        batch_mean, batch_var = tf.nn.moments(layer, [0], name='global_stats_'+name)
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        
        # PERFORM BATCH NORMALIZATION
        layer = batch_norm(layer,beta,gamma,phase_train,ema,batch_mean,batch_var)
        
        # ASSIGN PARAMETERS
        params = params+[beta,gamma,batch_mean,batch_var,ema]
        
    if actv==True:
        layer = tf.nn.relu(layer)
        
    return layer,params
    
def get_dense_layer_test_only_true(inp,params,actv):
    
    if len(params)==2:
        W,b = params
        bnorm = False
    elif len(params)==7:
        W,b,beta,gamma,batch_mean,batch_var,ema = params
        bnorm = True
    else:
        raise Exception("Wrong number of parameters")
        
    layer = dense(inp,W,b)
    
    if bnorm == True:
        layer = tf.nn.batch_normalization(layer,ema.average(batch_mean),ema.average(batch_var),beta,gamma,1e-3)
    if actv==True:
        layer = tf.nn.relu(layer)
    
    return layer
    
# THIS IS USED FOR FULL IMAGES ONLY
def get_dense_layer_test_only(inp,params,shape,actv=True):
    
    if len(params)==2:
        W,b = params
        bnorm = False
    elif len(params)==7:
        W,b,beta,gamma,batch_mean,batch_var,ema = params
        bnorm = True
    else:
        raise Exception("Wrong number of parameters")

    # RESHAPE WEIGHTS FOR CONVOLUTION
    previous_n_filters = int(W.shape.as_list()[0]/(shape[0]*shape[1]))
    new_shape = [shape[0],shape[1],previous_n_filters,W.shape.as_list()[1]]

    W = tf.reshape(W,new_shape)
    # CONVOLUTIONAL LAYER
    layer = conv2d(inp,W,b,1,padding="VALID")
    
    # BATCH NORMALIZATION
    if bnorm:
        layer = tf.nn.batch_normalization(layer,ema.average(batch_mean),ema.average(batch_var),beta,gamma,1e-3)
    
    # RELU LAYER
    if actv==True:
        layer = tf.nn.relu(layer)

    return layer

def dense(x,W,b):
    return tf.matmul(x,W)+b
    
def dropout(input_l,keep_prob):
    dropout_layer = tf.nn.dropout(input_l, keep_prob)
    return dropout_layer

"""
Auxiliary functions to get trainable parameters. These can be reused by setting
reuse = True.
"""

def get_params(shape, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable("W",shape,initializer=tf.random_normal_initializer(stddev=0.01))
        b = tf.get_variable("b",[shape[-1]],initializer=tf.constant_initializer())
    return W,b
    
def get_bn_params(filters, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        beta = tf.get_variable("beta",[filters],initializer=tf.constant_initializer())
        gamma = tf.get_variable("gamma",[filters],initializer=tf.constant_initializer(1.0))
    return beta,gamma
    

def batch_norm(x,beta,gamma,phase_train,ema,batch_mean,batch_var):
    
    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    
    mean,var = tf.cond(phase_train,mean_var_with_update,lambda: (ema.average(batch_mean),ema.average(batch_var)))
    
    normed = tf.nn.batch_normalization(x,mean,var,beta,gamma,1e-3)
    
    return normed
    
    
    
    
"""
ACTIVATIONS
 ***************************************************************************
"""
def relu(x):
    return tf.nn.relu(x)
    
def leaky_relu(x, alpha=0.1):
    return tf.maximum(x, alpha * x)

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def elu(x):
    return tf.nn.elu(x)
    
def activation(x, atype):
    
    if atype == "linear":
        return x
    if atype == "relu":
        return relu(x)
    if atype == "leaky_relu":
        return leaky_relu(x)
    
    if atype == "elu":
        return elu(x)
        
    if atype == "selu":
        return selu(x)
    
    raise Exception("Activation not understood: "+atype)
"""
 ***************************************************************************
"""    




"""

FULL IMAGE STUFF

"""

def strided(inp):
    k=2
    sh = inp.get_shape().as_list()
    strided = list()
    for i in range(sh[0]):
        strided.append(tf.strided_slice(inp,[i,0,0,0],[i+1,sh[1]-1,sh[2]-1,sh[3]],[1,k,k,1]))
        strided.append(tf.strided_slice(inp,[i,0,1,0],[i+1,sh[1]-1,sh[2],sh[3]],[1,k,k,1]))
        strided.append(tf.strided_slice(inp,[i,1,0,0],[i+1,sh[1],sh[2]-1,sh[3]],[1,k,k,1]))
        strided.append(tf.strided_slice(inp,[i,1,1,0],[i+1,sh[1],sh[2],sh[3]],[1,k,k,1]))

    return tf.concat(strided,axis=0)


def mult_unstrided(inp,times):
    if times==0:
        return inp
    else:
        times-=1
        return mult_unstrided(unstrided(inp),times)

def unstrided(inp):
    n_imgs = int(inp.shape[0]/4)
    out = np.zeros((n_imgs,inp.shape[1]*2,inp.shape[2]*2))
    for n in range(n_imgs):
        out[n,0:-1:2, 0:-1:2]=inp[n*4,:,:]
        out[n,0:-1:2, 1::2]=inp[n*4+1,:,:]
        out[n,1::2,   0:-1:2]=inp[n*4+2,:,:]
        out[n,1::2,   1::2]=inp[n*4+3,:,:]
    return out

def unstrided_special(inp):
    n_imgs = int(inp.shape[0]/4)
    out = np.zeros((n_imgs,inp.shape[1]*2,inp.shape[2]*2,inp.shape[3]))
    for n in range(n_imgs):
        out[n,0:-1:2, 0:-1:2,:]=inp[n*4,:,:,:]
        out[n,0:-1:2, 1::2,:]=inp[n*4+1,:,:,:]
        out[n,1::2,   0:-1:2,:]=inp[n*4+2,:,:,:]
        out[n,1::2,   1::2,:]=inp[n*4+3,:,:,:]
    return out


















