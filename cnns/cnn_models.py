# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 10:16:07 2017

@author: eduardo
"""
import cnns.cnn_lib as cnn_lib
import tensorflow as tf
import numpy as np


def detector36():
    """    
    conv1   3x3     32 filters    
    conv2   3x3     32 filters
    max_pool    
    
    conv3   3x3     128 filters
    conv4   3x3     128 filters
    max_pool
    
    dense1          256 filters
    dense2          256 filters
    output          2 filters
    """    
    
    
    model = Model([36,36])
    model.add_easy_layer(ltype="conv",filters_shape = [3,3], n_filters = 32)
    model.add_easy_layer(ltype="conv",filters_shape = [3,3], n_filters = 32)
    model.add_easy_layer(ltype="max_pool",k=2, stride=2)
    model.add_easy_layer(ltype="conv",filters_shape = [3,3], n_filters = 128)
    model.add_easy_layer(ltype="conv",filters_shape = [3,3], n_filters = 128)
    model.add_easy_layer(ltype="max_pool",k=2, stride=2)
    model.add_easy_layer(ltype="flatten")
    model.add_easy_layer(ltype="dense", n_filters = 256)
    model.add_easy_layer(ltype="dense", n_filters = 256)
    model.add_easy_layer(ltype="out", n_filters = 2)
    
    return model
    
    
def classifier156():
    model = Model([156,156])
    model.add_easy_layer(ltype="conv",filters_shape = [3,3], n_filters = 32)
    model.add_easy_layer(ltype="conv",filters_shape = [3,3], n_filters = 32)
    model.add_easy_layer(ltype="max_pool",k=2, stride=2)
    model.add_easy_layer(ltype="conv",filters_shape = [3,3], n_filters = 64)
    model.add_easy_layer(ltype="conv",filters_shape = [3,3], n_filters = 64)
    model.add_easy_layer(ltype="max_pool",k=2, stride=2)
    model.add_easy_layer(ltype="conv",filters_shape = [3,3], n_filters = 128)
    model.add_easy_layer(ltype="conv",filters_shape = [3,3], n_filters = 128)
    model.add_easy_layer(ltype="max_pool",k=2, stride=2)
    model.add_easy_layer(ltype="conv",filters_shape = [3,3], n_filters = 128)
    model.add_easy_layer(ltype="conv",filters_shape = [3,3], n_filters = 128)
    model.add_easy_layer(ltype="max_pool",k=2, stride=2)
    model.add_easy_layer(ltype="flatten")
    model.add_easy_layer(ltype="dense", n_filters = 256)
    model.add_easy_layer(ltype="dense", n_filters = 256)
    model.add_easy_layer(ltype="out", n_filters = 2)
    
def test_model():
    model = Model([36,36])
    model.add_easy_layer(ltype="conv", filters_shape = [36,36], n_filters = 32)
    model.add_easy_layer(ltype="flatten")
    model.add_easy_layer(ltype="out",n_filters=2)
    model._compile()
    return model
    
def alexMet(n_classes):
    
    alex = Model([224,224])
    
    alex.add_easy_layer(ltype = "conv",filters_shape = [11,11],n_filters = 32,padding="SAME",stride=4,activation="relu")
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 32, moments=[0,1,2])
    alex.add_easy_layer(ltype="max_pool",k=2, stride=2)
    alex.add_easy_layer(ltype = "conv",filters_shape = [5,5],n_filters = 64,padding="SAME",stride=1,activation="relu")
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 64, moments=[0,1,2])    
    alex.add_easy_layer(ltype="max_pool",k=2, stride=2)
    alex.add_easy_layer(ltype = "conv",filters_shape = [3,3],n_filters = 128,padding="SAME",stride=1,activation="relu")
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 128, moments=[0,1,2])
    alex.add_easy_layer(ltype = "conv",filters_shape = [3,3],n_filters = 128,padding="SAME",stride=1,activation="relu")
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 128, moments=[0,1,2])
    alex.add_easy_layer(ltype = "conv",filters_shape = [3,3],n_filters = 128,padding="SAME",stride=1,activation="relu")
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 128, moments=[0,1,2])
    alex.add_easy_layer(ltype="flatten")
    alex.add_easy_layer(ltype="dense", n_filters = 256)
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 256, moments=[0])
    alex.add_easy_layer(ltype="dense", n_filters = 256)
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 256, moments=[0])
    alex.add_easy_layer(ltype="out", n_filters = n_classes)
    
    alex._compile()
    print([x.out for x in alex.layers])
    
    return alex
    
def alexMet_rotat(n_classes):
    
    alex = Model([224,224])
    
    alex.add_easy_layer(ltype = "conv_rotat",filters_shape = [11,11],n_filters = 32,padding="SAME",stride=4,activation="relu")
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 32, moments=[0,1,2])
    alex.add_easy_layer(ltype="max_pool",k=2, stride=2)
    alex.add_easy_layer(ltype = "conv_rotat",filters_shape = [5,5],n_filters = 64,padding="SAME",stride=1,activation="relu")
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 64, moments=[0,1,2])    
    alex.add_easy_layer(ltype="max_pool",k=2, stride=2)
    alex.add_easy_layer(ltype = "conv_rotat",filters_shape = [3,3],n_filters = 128,padding="SAME",stride=1,activation="relu")
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 128, moments=[0,1,2])
    alex.add_easy_layer(ltype = "conv_rotat",filters_shape = [3,3],n_filters = 128,padding="SAME",stride=1,activation="relu")
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 128, moments=[0,1,2])
    alex.add_easy_layer(ltype = "conv_rotat",filters_shape = [3,3],n_filters = 128,padding="SAME",stride=1,activation="relu")
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 128, moments=[0,1,2])
    alex.add_easy_layer(ltype="flatten")
    alex.add_easy_layer(ltype="dense", n_filters = 256)
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 256, moments=[0])
    alex.add_easy_layer(ltype="dense", n_filters = 256)
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 256, moments=[0])
    alex.add_easy_layer(ltype="out", n_filters = n_classes)
    #alex.add_easy_layer(ltype = "conv_rotat",filters_shape = [14,14],n_filters = 256,padding="VALID",stride=1,activation="relu")
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 256, moments=[0,1,2])    
    #alex.add_easy_layer(ltype="flatten")
    #alex.add_easy_layer(ltype="dense", n_filters = 256)
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 256, moments=[0])
    #alex.add_easy_layer(ltype="out", n_filters = n_classes)
    
    alex._compile()
    print([x.out for x in alex.layers])
    return alex
    


def alexMet_first_dense(n_classes):
    
    alex = Model([224,224])
    
    alex.add_easy_layer(ltype = "conv",filters_shape = [11,11],n_filters = 32,padding="SAME",stride=4,activation="relu")
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 32, moments=[0,1,2])
    alex.add_easy_layer(ltype="max_pool",k=2, stride=2)
    alex.add_easy_layer(ltype = "conv",filters_shape = [5,5],n_filters = 64,padding="SAME",stride=1,activation="relu")
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 64, moments=[0,1,2])    
    alex.add_easy_layer(ltype="max_pool",k=2, stride=2)
    alex.add_easy_layer(ltype = "conv",filters_shape = [3,3],n_filters = 128,padding="SAME",stride=1,activation="relu")
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 128, moments=[0,1,2])
    alex.add_easy_layer(ltype = "conv",filters_shape = [3,3],n_filters = 128,padding="SAME",stride=1,activation="relu")
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 128, moments=[0,1,2])
    alex.add_easy_layer(ltype = "conv",filters_shape = [3,3],n_filters = 128,padding="SAME",stride=1,activation="relu")
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 128, moments=[0,1,2])
    alex.add_easy_layer(ltype = "conv_rotat",filters_shape = [14,14],n_filters = 256,padding="VALID",stride=1,activation="relu")
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 256, moments=[0,1,2])    
    alex.add_easy_layer(ltype="flatten")
    alex.add_easy_layer(ltype="dense", n_filters = 256)
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 256, moments=[0])
    alex.add_easy_layer(ltype="out", n_filters = n_classes)
    #alex.add_easy_layer(ltype = "conv_rotat",filters_shape = [14,14],n_filters = 256,padding="VALID",stride=1,activation="relu")
   # alex.add_easy_layer(ltype = "batchnorm",n_filters = 256, moments=[0,1,2])    
    #alex.add_easy_layer(ltype="flatten")
    #alex.add_easy_layer(ltype="dense", n_filters = 256)
    #alex.add_easy_layer(ltype = "batchnorm",n_filters = 256, moments=[0])
    #alex.add_easy_layer(ltype="out", n_filters = n_classes)
    
    alex._compile()
    print([x.out for x in alex.layers])
    return alex


def mnist_general_arch(n_filters=4, **args):
    
    reg_param_vars = args.get("reg_param_vars")
    reg_param_weights = args.get("reg_param_weights")
    rotat = args.get("rotat")
    
    model = Model([28,28])
    model.add_easy_layer(ltype="conv",filters_shape = [5,5],n_filters = 4*n_filters, padding="SAME",stride=1,activation="relu")
    model.add_easy_layer(ltype="conv",filters_shape = [3,3],n_filters = 8*n_filters, padding="SAME",stride=1,activation="relu")
    model.add_easy_layer(ltype="max_pool",k=2, stride=2)
    
    model.add_easy_layer(ltype="conv",filters_shape = [3,3],n_filters = 16*n_filters, padding="SAME",stride=1,activation="relu")    
    model.add_easy_layer(ltype="max_pool",k=2, stride=2)
    #model.add_easy_layer(ltype = "batchnorm",n_filters = 64, moments=[0,1,2])    
    
    if rotat:        
        model.add_easy_layer(ltype="conv_rotatV2",filters_shape = [7,7],n_filters = 64*n_filters, padding="VALID",stride=1,activation="linear")
        model.add_easy_layer(ltype="flatten")
        #model.add_easy_layer(ltype = "batchnorm",n_filters = 1024, moments=[0]) 
    else:
        model.add_easy_layer(ltype="flatten")
        model.add_easy_layer(ltype="dense", n_filters = 64*n_filters)
        #model.add_easy_layer(ltype = "batchnorm",n_filters = 256, moments=[0])
    
    model.add_easy_layer(ltype="dense",n_filters = 256)
    model.add_easy_layer(ltype="out", n_filters = 10)
    
    
    
    if reg_param_vars!=None:
        print("added variational loss")
        reshaped_layer = tf.reshape(model.layers[6].out,shape=[-1,4,1024])
        _,add_loss = tf.nn.moments(reshaped_layer,axes = [1])
        add_loss = tf.reduce_mean(add_loss)
    
        
        model._compile(add_loss = add_loss*reg_param_vars)  
    else:
        print("normal model")
        model._compile()
    
    return model
    #reg_param_vars, reg_param_weights, conv_constraints):
    

def mnist_soft_inv_arch(reg_param = 0.01,variations = 4):
    
    n_base_filter = 4
    
    model = Model([28,28])
    model.add_easy_layer(ltype="conv",filters_shape = [5,5],n_filters = 4*n_base_filter, padding="SAME",stride=1,activation="relu")
    model.add_easy_layer(ltype="conv",filters_shape = [3,3],n_filters = 8*n_base_filter, padding="SAME",stride=1,activation="relu")
    model.add_easy_layer(ltype="max_pool",k=2, stride=2)
    
    model.add_easy_layer(ltype="conv",filters_shape = [3,3],n_filters = 16*n_base_filter, padding="SAME",stride=1,activation="relu")    
    model.add_easy_layer(ltype="max_pool",k=2, stride=2)
    #model.add_easy_layer(ltype = "batchnorm",n_filters = 64, moments=[0,1,2])    
    
    model.add_easy_layer(ltype="flatten")
    model.add_easy_layer(ltype="dense", n_filters = 1024)
    #model.add_easy_layer(ltype = "batchnorm",n_filters = 1024, moments=[0]) 

    model.add_easy_layer(ltype="dense",n_filters = 256)
    model.add_easy_layer(ltype="out", n_filters = 10)  


    reshaped_layer = tf.reshape(model.layers[6].out,shape=[-1,4,1024])
    _,add_loss = tf.nn.moments(reshaped_layer,axes = [1])
    add_loss = tf.reduce_mean(add_loss)
    
        
    model._compile(add_loss = add_loss*reg_param)
        
    return model
    
def mnist_arch(rotat=False,v1=False):
    
    n_base_filter = 4
    
    model = Model([28,28])
    model.add_easy_layer(ltype="conv",filters_shape = [5,5],n_filters = 4*n_base_filter, padding="SAME",stride=1,activation="relu")
    model.add_easy_layer(ltype="conv",filters_shape = [3,3],n_filters = 8*n_base_filter, padding="SAME",stride=1,activation="relu")
    model.add_easy_layer(ltype="max_pool",k=2, stride=2)
    
    model.add_easy_layer(ltype="conv",filters_shape = [3,3],n_filters = 16*n_base_filter, padding="SAME",stride=1,activation="relu")    
    model.add_easy_layer(ltype="max_pool",k=2, stride=2)
    #model.add_easy_layer(ltype = "batchnorm",n_filters = 64, moments=[0,1,2])    
    
    if rotat==False:        
        model.add_easy_layer(ltype="flatten")
        model.add_easy_layer(ltype="dense", n_filters = 1024)
        #model.add_easy_layer(ltype = "batchnorm",n_filters = 1024, moments=[0]) 
    elif rotat==True:
        if v1:
            model.add_easy_layer(ltype="conv_rotat",filters_shape = [7,7],n_filters = 1024, padding="VALID",stride=1,activation="linear")
        else:
            model.add_easy_layer(ltype="conv_rotatV2",filters_shape = [7,7],n_filters = 1024, padding="VALID",stride=1,activation="linear")
        model.add_easy_layer(ltype="flatten")
        #model.add_easy_layer(ltype = "batchnorm",n_filters = 256, moments=[0])
    else:
          return
    model.add_easy_layer(ltype="dense",n_filters = 256)
    model.add_easy_layer(ltype="out", n_filters = 10)
    
    model._compile()
    
    return model
    
    
    
class Model():
    def __init__(self, inp_shape, in_channels=1):
        self.inp_shape = inp_shape
        self.in_channels = in_channels
        self.inp = tf.placeholder(tf.float32,[None,*inp_shape,in_channels])
        self.phase_train = tf.placeholder(tf.bool,[])
        self.keep_prob = tf.placeholder(tf.float32,[])
        self.y = tf.placeholder(tf.int32,[None])
        self.out = self.inp
        self.out_shape = [inp_shape[0],inp_shape[1],self.in_channels]
        self.layers = []
        
    def add_easy_layer(self,**params):
        layer = easy_layer(self,**params)
        self.layers.append(layer)
        self.out = layer.out
        return
        
    def _compile(self,add_loss = None):
        self.pred = tf.nn.softmax(self.out)
        
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.out))
        
        if add_loss != None:
            self.loss = self.loss+add_loss
        #self.acc = tf.metrics.accuracy(labels=self.y,logits=self.out)
        self.learning_rate = tf.placeholder(tf.float32,[])
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_step
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()
        return
    
    def add_additional_loss(self,loss):
        self.loss = self.loss+loss
        
    def train(self,sess,batchx,batchy,learning_rate):
        outs = sess.run([self.loss,self.train_op,self.pred], feed_dict={self.inp:batchx, self.y:batchy, self.phase_train:True, 
                                           self.keep_prob:0.8, self.learning_rate:learning_rate})
        return outs[0],outs[2]
        
    def test(self,sess,batchx,batchy):
        return sess.run([self.loss, self.pred],feed_dict={self.inp:batchx, self.y:batchy, self.phase_train:False, self.keep_prob:1})
    
    def test_loss_acc(self,sess,batchx,batchy):
        return 0
        
    def print_layers(self):
        for l in self.layers:
            print(l)
            
    def save(self,sess,path):
        self.saver.save(sess,path)
        
    def load(self,sess, path):
        self.saver.restore(sess,path)
        

def easy_layer(model, **args):
    
    ltype = args.get("ltype")
    
    if ltype == "conv":
        n_filters =         args.get("n_filters")
        filters_shape =     args.get("filters_shape")
        padding =           args.get("padding")
        stride =            args.get("stride")
        activation =        args.get("activation")
        shape =             [filters_shape[0],filters_shape[1],model.out_shape[2],n_filters]
        name =              "conv"+str(len(model.layers))
        
        layer = Convolutional_Layer(shape, name , model.out, padding=padding, stride=stride, activation=activation)
        
        #model.out_shape[0] = model.out_shape[0]-int((filters_shape[0]-1))
        #model.out_shape[1] = model.out_shape[1]-int((filters_shape[1]-1))
        #model.out_shape[2] = n_filters
        model.out_shape = layer.out.shape.as_list()[1:4]
        
        return layer
        
    if ltype == "conv_rotat":
        n_filters =         args.get("n_filters")
        filters_shape =     args.get("filters_shape")
        padding =           args.get("padding")
        stride =            args.get("stride")
        activation =        args.get("activation")
        shape = [filters_shape[0],filters_shape[1],model.out_shape[2],n_filters]
        name = "conv_rotat"+str(len(model.layers))
        
        layer = Convolutional_Layer_Rotat(shape, name , model.out, padding=padding, stride=stride, activation=activation)
        
        model.out_shape = layer.out.shape.as_list()[1:4]
        
        return layer
        
    if ltype == "conv_rotatV2":
        n_filters =         args.get("n_filters")
        filters_shape =     args.get("filters_shape")
        padding =           args.get("padding")
        stride =            args.get("stride")
        activation =        args.get("activation")
        shape = [filters_shape[0],filters_shape[1],model.out_shape[2],n_filters]
        name = "conv_rotat"+str(len(model.layers))
        
        layer = Convolutional_Layer_RotatV2(shape, name , model.out, padding=padding, stride=stride, activation=activation)
        
        model.out_shape = layer.out.shape.as_list()[1:4]
        
        return layer
        
    if ltype == "max_pool":#make a class_wrapper
        k = args.get("k")
        stride = args.get("stride")
        layer = cnn_lib.maxpool2d(model.out, k, stride)
        model.out_shape[0] = int(model.out_shape[0]/2)
        model.out_shape[1] = int(model.out_shape[1]/2)
        
        return Generic_Layer(layer)
        
    if ltype == "flatten":#make a class_wrapper
        layer = cnn_lib.flatten(model.out)
        model.out_shape = [model.out_shape[0]*model.out_shape[1]*model.out_shape[2]]        
        return Generic_Layer(layer)
                
    if ltype == "dense":
        n_filters =         args.get("n_filters")
        activation =        args.get("activation")
        
        layer = Dense_Layer([model.out_shape[0],n_filters], "dense"+str(len(model.layers)), model.out,activation=activation)
        model.out_shape[0] = n_filters
        return layer
        
    if ltype == "denseV2":
        n_filters =         args.get("n_filters")
        activation =        args.get("activation")
        
        layer = Dense_LayerV2([model.out_shape[0],n_filters], "dense"+str(len(model.layers)), model.out,activation=activation)
        model.out_shape[0] = n_filters//4
        
        return layer
        
    if ltype == "out":
        n_filters = args.get("n_filters")
        layer = Dense_Layer([model.out_shape[0],n_filters], "out"+str(len(model.layers)), model.out, activation="linear")
        model.out_shape[0] = n_filters
        return layer
        
    if ltype == "batchnorm":
        n_filters = args.get("n_filters")
        name = "bn"+str(len(model.layers))
        phase_train = model.phase_train
        moments = args.get("moments")
        layer = Batchnorm_Layer(n_filters,name,model.out,phase_train,moments)
        return layer
        
"""
    BETTER LIB FOR CONVNETS (MORE GENERAL)
    TODO EXP_MOVING_WEIGHTS
"""

class Generic_Layer():# MAX_POOL, FLATTEN, ...
    def __init__(self,layer):
        self.out = layer
        
class Convolutional_Layer():
    def __init__(self, shape, name, inp, padding = None , stride = None, activation = None):

        padding = "VALID" if padding == None else padding
        stride = 1 if stride == None else stride    
        activation = "relu" if activation == None else activation 
            
        self.name = name
        self.shape = shape
        
        with tf.variable_scope(name) as scope:       
            self.W = Param(shape,"W",initializer("normal",np.sqrt(2/(shape[0]*shape[1]*shape[3]))))
            self.b = Param(shape[3],"b",initializer("constant",0.0))
            self.out = cnn_lib.conv2d(inp,self.W.value,self.b.value,stride=stride,padding=padding)            
            self.out = cnn_lib.activation(self.out,activation)
            
        
class Convolutional_Layer_Rotat():
    def __init__(self, shape, name, inp, padding = None , stride = None, activation = None):
        
        padding = "VALID" if padding == None else padding
        stride = 1 if stride == None else stride    
        activation = "relu" if activation == None else activation 
        
        self.name = name
        self.shape = shape
        W_shape = shape.copy()
        W_shape[3]=int(W_shape[3]/4)
        with tf.variable_scope(name) as scope:       
            self.W = Param(W_shape,"W",initializer("normal",np.sqrt(2/(shape[0]*shape[1]*shape[3]))))
            self.b = Param(W_shape[3],"b",initializer("constant",0.0))
            
            layers = []
            W = self.W.value
            for i in range(4):
                layers.append(cnn_lib.conv2d(inp,W,self.b.value,stride=stride,padding=padding))
                W = cnn_lib.rotate(W)
                #self.out = cnn_lib.conv2d(inp,self.W.value,self.b.value,stride=stride,padding=padding) 
            layer = tf.concat(layers,axis = 3)
            self.out = cnn_lib.activation(layer,activation)
            
class Convolutional_Layer_RotatV2():
    def __init__(self, shape, name, inp, padding = None , stride = None, activation = None):
        
        padding = "VALID" if padding == None else padding
        stride = 1 if stride == None else stride    
        activation = "relu" if activation == None else activation 
        
        self.name = name
        self.shape = shape
        W_shape = shape.copy()
        W_shape[3]=int(W_shape[3]/4)
        with tf.variable_scope(name) as scope:       
            self.W = Param(W_shape,"W",initializer("normal",np.sqrt(2/(shape[0]*shape[1]*shape[3]))))
            self.b = Param(W_shape[3],"b",initializer("constant",0.0))
            
            layers = []
            W = self.W.value
            for i in range(4):
                layers.append(cnn_lib.conv2d(inp,W,self.b.value,stride=stride,padding=padding))
                W = cnn_lib.rotate(W)
                #self.out = cnn_lib.conv2d(inp,self.W.value,self.b.value,stride=stride,padding=padding) 
            layer = tf.stack(layers,axis = 4)
            layer = tf.reduce_max(layer,axis=4)            
            self.out = cnn_lib.activation(layer,activation)
        
        
class Dense_Layer():
    def __init__(self, shape, name, inp, activation = None):
        
        activation = "relu" if activation == None else activation 
        
        self.name = name
        self.shape = shape
        
        with tf.variable_scope(name) as scope:       
            self.W = Param(shape,"W",initializer("normal",np.sqrt(2/shape[0])))
            self.b = Param(shape[1],"b",initializer("constant",0.0))
            self.out = cnn_lib.dense(inp,self.W.value,self.b.value)
            self.out = cnn_lib.activation(self.out,activation)


class Dense_LayerV2():
    def __init__(self, shape, name, inp, activation = None):
        
        activation = "relu" if activation == None else activation 
        
        self.name = name
        self.shape = shape
        
        with tf.variable_scope(name) as scope:       
            self.W = Param(shape,"W",initializer("normal",np.sqrt(2/shape[0])))
            self.b = Param(shape[1],"b",initializer("constant",0.0))
            layer = cnn_lib.dense(inp,self.W.value,self.b.value)
            
            self.out = tf.reduce_max(tf.reshape(layer,[-1,shape[1]//4,4]),axis=2)
            self.out = cnn_lib.activation(self.out,activation)


class Batchnorm_Layer():
     def __init__(self, channels, name, inp, phase_train ,moments=[0,1,2]):  
         
         with tf.variable_scope(name) as scope:
             self.gamma = Param([channels],"gamma",initializer("constant",1.0)) 
             self.beta = Param([channels],"beta",initializer("constant",0.0))
             
             self.ema = tf.train.ExponentialMovingAverage(decay=0.8)
             self.batch_mean, self.batch_var = tf.nn.moments(inp,moments)
             
             def mean_var_with_update():
                ema_apply_op = self.ema.apply([self.batch_mean, self.batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(self.batch_mean), tf.identity(self.batch_var)
            
             mean,var = tf.cond(phase_train,mean_var_with_update,lambda: (self.ema.average(self.batch_mean),
                                                                         self.ema.average(self.batch_var)))
            
         self.out = tf.nn.batch_normalization(inp,mean,var,self.beta.value,self.gamma.value,1e-3)
             
    
class Param():
    def __init__(self, shape, name, initializer):
        self.value = tf.get_variable(name,shape,initializer=initializer)
        self.shape = shape
        self.name = name
        
        
def initializer(i_type,param):
    if i_type == "normal":
        return tf.random_normal_initializer(stddev=param)
    elif i_type == "constant":
        return tf.constant_initializer(param)
 

#alex = alexMet_rotat(3)



"""
# CLASS FOR A TYPICAL CNN MODEL
class Model():
    
    def __init__(self):
        self.layers = []
        self.inp = None
        self.out = None
        
    def add(self,layer):
        self.layers.append(layer)
        self.out = layer.op
        
    def save(self,name,function):
        return

        
class Layer():
    
    def __init__(self, operation , parameters=[]):
        self.op = operation
        self.params = parameters
    
"""      
"""
def detector36(x,phase_train,keep_prob):
   


    detector = Model()
    detector.inp = x
    
    layer, params = cnn_lib.get_conv_layer("conv1",detector.inp,[3,3,1,32],phase_train)
    detector.add(Layer(layer,params))
    layer, params = cnn_lib.get_conv_layer("conv2",detector.out,[3,3,32,32],phase_train)
    detector.add(Layer(layer,params))
    layer = cnn_lib.maxpool2d(detector.out,k=2)
    detector.add(Layer(layer,[]))
    
    layer, params = cnn_lib.get_conv_layer("conv3",detector.out,[3,3,32,128],phase_train)    
    detector.add(Layer(layer,params))
    layer, params = cnn_lib.get_conv_layer("conv4",detector.out,[3,3,128,128],phase_train)    
    detector.add(Layer(layer,params))    
    layer = cnn_lib.maxpool2d(detector.out,k=2)
    detector.add(Layer(layer,[]))    
    
    layer = Layer(cnn_lib.flatten(detector.out),[])
    layer.flatten_dim = [6,6]
    detector.add(layer)
    
    layer,params = cnn_lib.get_dense_layer("dense1",detector.out,[128*6*6,256],phase_train,actv=True)    
    detector.add(Layer(layer,params))
    layer = cnn_lib.dropout(layer, keep_prob)
    detector.add(Layer(layer,[]))    
    layer,params = cnn_lib.get_dense_layer("dense2",detector.out,[256,256],phase_train,actv=True)        
    detector.add(Layer(layer,params))
    layer = cnn_lib.dropout(layer, keep_prob)
    detector.add(Layer(layer,[]))
    layer,params = cnn_lib.get_dense_layer("out",detector.out,[256,2],False,actv=False)    
    detector.add(Layer(layer,params))
    
    return detector


def detector36_ema_weights(test_x ,model):
    
    ema_model = Model()
    ema_model.inp = test_x
    
    exp_keeper = tf.train.ExponentialMovingAverage(0.99)
    exp_keeper_ops = []
    
    # FIRST CONV LAYER    
    W,b,beta,gamma,batch_mean,batch_var,ema = model.layers[0].params    
    exp_keeper_ops = exp_keeper_ops+[exp_keeper.apply([W, b, beta, gamma])]
    params = [exp_keeper.average(W),    exp_keeper.average(b),
              exp_keeper.average(beta), exp_keeper.average(gamma),
              batch_mean,               batch_var,              ema]
    layer = cnn_lib.get_conv_layer_test_only(ema_model.inp,params)
    ema_model.add(Layer(layer,params))
    
    # SECOND CONV LAYER
    W,b,beta,gamma,batch_mean,batch_var,ema = model.layers[1].params
    exp_keeper_ops = exp_keeper_ops+[exp_keeper.apply([W, b, beta, gamma])]
    params = [exp_keeper.average(W),    exp_keeper.average(b),
              exp_keeper.average(beta), exp_keeper.average(gamma),
              batch_mean,               batch_var,              ema]
    layer = cnn_lib.get_conv_layer_test_only(ema_model.out,params)
    ema_model.add(Layer(layer,params))
    
    layer = cnn_lib.maxpool2d(ema_model.out,k=2)
    ema_model.add(Layer(layer,[]))
    
    # THIRD CONV LAYER
    W,b,beta,gamma,batch_mean,batch_var,ema = model.layers[3].params
    exp_keeper_ops = exp_keeper_ops+[exp_keeper.apply([W, b, beta, gamma])]
    params = [exp_keeper.average(W),    exp_keeper.average(b),
              exp_keeper.average(beta), exp_keeper.average(gamma),
              batch_mean,               batch_var,              ema]
    layer = cnn_lib.get_conv_layer_test_only(ema_model.out,params)
    ema_model.add(Layer(layer,params))
    
    # FORTH CONV LAYER
    W,b,beta,gamma,batch_mean,batch_var,ema = model.layers[4].params
    exp_keeper_ops = exp_keeper_ops+[exp_keeper.apply([W, b, beta, gamma])]
    params = [exp_keeper.average(W),    exp_keeper.average(b),
              exp_keeper.average(beta), exp_keeper.average(gamma),
              batch_mean,               batch_var,              ema]
    layer = cnn_lib.get_conv_layer_test_only(ema_model.out,params)
    ema_model.add(Layer(layer,params))
    
    layer = cnn_lib.maxpool2d(ema_model.out,k=2)
    ema_model.add(Layer(layer,[]))
    
    layer = Layer(cnn_lib.flatten(ema_model.out),[])
    layer.flatten_dim = [6,6]
    ema_model.add(layer)    
    
    # FIRST DENSE LAYER
    W,b,beta,gamma,batch_mean,batch_var,ema = model.layers[7].params
    exp_keeper_ops = exp_keeper_ops+[exp_keeper.apply([W, b, beta, gamma])]
    params = [exp_keeper.average(W),    exp_keeper.average(b),
              exp_keeper.average(beta), exp_keeper.average(gamma),
              batch_mean,               batch_var,              ema]
    layer = cnn_lib.get_dense_layer_test_only_true(ema_model.out,   params,  True)
    ema_model.add(Layer(layer,params))
    
    layer = cnn_lib.dropout(layer, 1.0)
    ema_model.add(Layer(layer,[]))    

    # SECOND DENSE LAYER
    W,b,beta,gamma,batch_mean,batch_var,ema = model.layers[9].params
    exp_keeper_ops = exp_keeper_ops+[exp_keeper.apply([W, b, beta, gamma])]
    params = [exp_keeper.average(W),    exp_keeper.average(b),
              exp_keeper.average(beta), exp_keeper.average(gamma),
              batch_mean,               batch_var,              ema]
    layer = cnn_lib.get_dense_layer_test_only_true(ema_model.out,   params,  True)
    ema_model.add(Layer(layer,params))
    
    layer = cnn_lib.dropout(layer, 1.0)
    ema_model.add(Layer(layer,[]))    

    # FINAL LAYER
    W,b = model.layers[11].params    
    exp_keeper_ops = exp_keeper_ops+[exp_keeper.apply([W, b])]
    params = [exp_keeper.average(W),    exp_keeper.average(b)]
    layer = cnn_lib.get_dense_layer_test_only_true(ema_model.out,  params, False)
    ema_model.add(Layer(layer,params))

    maintain_averages_op = tf.group(*exp_keeper_ops)

    return ema_model,maintain_averages_op

def l2_loss(model,lamb):
    var = []
    for layer in model.layers:
        if len(layer.params)>0:
            var.append(layer.params[0])
    
    loss = tf.add_n([tf.nn.l2_loss(v) for v in var])
    return loss*lamb
    
    
    

THIS IS A TEST ONLY NET


def get_full_image_detector(model, x ):
    
    f_det = Model()
    f_det.inp = x
            
    layer = cnn_lib.get_conv_layer_test_only(f_det.inp,model.layers[0].params)
    f_det.add(Layer(layer))
    layer = cnn_lib.get_conv_layer_test_only(f_det.out,model.layers[1].params)    
    f_det.add(Layer(layer))
    
    layer = cnn_lib.maxpool2d(f_det.out,k=2,stride=1)
    layer = cnn_lib.strided(layer)
    f_det.add(Layer(layer))
    
    layer = cnn_lib.get_conv_layer_test_only(f_det.out,model.layers[3].params)
    f_det.add(Layer(layer))
    layer = cnn_lib.get_conv_layer_test_only(f_det.out,model.layers[4].params)    
    f_det.add(Layer(layer))
    layer = cnn_lib.maxpool2d(f_det.out,k=2,stride=1)
    layer = cnn_lib.strided(layer)
    f_det.add(Layer(layer))
    
    layer = cnn_lib.get_dense_layer_test_only(f_det.out,model.layers[7].params,[6,6],actv=True)
    f_det.add(Layer(layer))
    layer = cnn_lib.dropout(layer, 1.0)
    f_det.add(Layer(layer))    
    layer = cnn_lib.get_dense_layer_test_only(f_det.out,model.layers[9].params,[1,1],actv=True)
    f_det.add(Layer(layer))
    layer = cnn_lib.dropout(layer, 1.0)
    f_det.add(Layer(layer))    
    layer = cnn_lib.get_dense_layer_test_only(f_det.out,model.layers[11].params,[1,1],actv=False)
    f_det.add(Layer(layer))
    
    return f_det
    
    
    
def get_full_image_layer(layer, inp, shape=None):
    
    if layer.phase_train != None:
        phase_train = "test_only"
    else:
        phase_train = None

    if layer.ltype=="conv":
        return Layer(layer.ltype,inp,layer.shape,actv=layer.actv,phase_train = phase_train ,name=layer.name[4::],reuse=True)
    
    if layer.ltype=="max_pool":
        return Layer("max_pool_strided", inp)
    
    if layer.ltype=="dense":
        if shape!=None:
            assert shape[2]==layer.shape[0]
            shape=shape+[layer.shape[-1]]
        else:
            shape=[1,1]+layer.shape
        return Layer("dense_to_conv", inp, shape, actv=layer.actv, phase_train = phase_train, name=layer.name[5::], reuse=True)
    
    

    
    
    
def detector76(x,phase_train,keep_prob):
    
    conv1   3x3     32 filters    
    conv2   3x3     32 filters
    max_pool    
    
    conv3   3x3     32 filters    
    conv4   3x3     32 filters
    max_pool 
    
    conv5   3x3     128 filters
    conv6   3x3     128 filters
    max_pool
    
    dense1          256 filters
    dense2          256 filters
    output          2 filters


    detector = Model()
    detector.inp = x
    
    layer, params = cnn_lib.get_conv_layer("conv1",detector.inp,[3,3,1,32],phase_train)
    detector.add(Layer(layer,params))
    layer, params = cnn_lib.get_conv_layer("conv2",detector.out,[3,3,32,32],phase_train)
    detector.add(Layer(layer,params))
    layer = cnn_lib.maxpool2d(detector.out,k=2)
    detector.add(Layer(layer,[]))
    
    layer, params = cnn_lib.get_conv_layer("conv3",detector.out,[3,3,32,64],phase_train)    
    detector.add(Layer(layer,params))
    layer, params = cnn_lib.get_conv_layer("conv4",detector.out,[3,3,64,64],phase_train)    
    detector.add(Layer(layer,params))    
    layer = cnn_lib.maxpool2d(detector.out,k=2)
    detector.add(Layer(layer,[]))
    
    layer, params = cnn_lib.get_conv_layer("conv5",detector.out,[3,3,64,128],phase_train)    
    detector.add(Layer(layer,params))
    layer, params = cnn_lib.get_conv_layer("conv6",detector.out,[3,3,128,128],phase_train)    
    detector.add(Layer(layer,params))    
    layer = cnn_lib.maxpool2d(detector.out,k=2)
    detector.add(Layer(layer,[]))
    
    layer = Layer(cnn_lib.flatten(detector.out),[])
    layer.flatten_dim = [6,6]
    detector.add(layer)
    
    layer,params = cnn_lib.get_dense_layer("dense1",detector.out,[128*6*6,256],phase_train,actv=True)    
    detector.add(Layer(layer,params))
    layer = cnn_lib.dropout(layer, keep_prob)
    detector.add(Layer(layer,[]))    
    layer,params = cnn_lib.get_dense_layer("dense2",detector.out,[256,256],phase_train,actv=True)        
    detector.add(Layer(layer,params))
    layer = cnn_lib.dropout(layer, keep_prob)
    detector.add(Layer(layer,[]))
    layer,params = cnn_lib.get_dense_layer("out",detector.out,[256,2],False,actv=False)    
    detector.add(Layer(layer,params))
    
    return detector
    

def get_full_image_detector76(model, x ):
    
    f_det = Model()
    f_det.inp = x
            
    layer = cnn_lib.get_conv_layer_test_only(f_det.inp,model.layers[0].params)
    f_det.add(Layer(layer))
    layer = cnn_lib.get_conv_layer_test_only(f_det.out,model.layers[1].params)    
    f_det.add(Layer(layer))
    
    layer = cnn_lib.maxpool2d(f_det.out,k=2,stride=1)
    layer = cnn_lib.strided(layer)
    f_det.add(Layer(layer))
    
    layer = cnn_lib.get_conv_layer_test_only(f_det.out,model.layers[3].params)
    f_det.add(Layer(layer))
    layer = cnn_lib.get_conv_layer_test_only(f_det.out,model.layers[4].params)    
    f_det.add(Layer(layer))
    layer = cnn_lib.maxpool2d(f_det.out,k=2,stride=1)
    layer = cnn_lib.strided(layer)
    f_det.add(Layer(layer))
    
    layer = cnn_lib.get_conv_layer_test_only(f_det.out,model.layers[6].params)
    f_det.add(Layer(layer))
    layer = cnn_lib.get_conv_layer_test_only(f_det.out,model.layers[7].params)    
    f_det.add(Layer(layer))
    
    layer = cnn_lib.maxpool2d(f_det.out,k=2,stride=1)
    layer = cnn_lib.strided(layer)
    f_det.add(Layer(layer))
    
    layer = cnn_lib.get_dense_layer_test_only(f_det.out,model.layers[10].params,[6,6],actv=True)
    f_det.add(Layer(layer))
    layer = cnn_lib.dropout(layer, 1.0)
    f_det.add(Layer(layer))    
    layer = cnn_lib.get_dense_layer_test_only(f_det.out,model.layers[12].params,[1,1],actv=True)
    f_det.add(Layer(layer))
    layer = cnn_lib.dropout(layer, 1.0)
    f_det.add(Layer(layer))    
    layer = cnn_lib.get_dense_layer_test_only(f_det.out,model.layers[14].params,[1,1],actv=False)
    f_det.add(Layer(layer))
    
    return f_det

def detector76_ema_weights(test_x ,model):
    
    ema_model = Model()
    ema_model.inp = test_x
    
    exp_keeper = tf.train.ExponentialMovingAverage(0.99)
    exp_keeper_ops = []
    
    # FIRST CONV LAYER    
    W,b,beta,gamma,batch_mean,batch_var,ema = model.layers[0].params    
    exp_keeper_ops = exp_keeper_ops+[exp_keeper.apply([W, b, beta, gamma])]
    params = [exp_keeper.average(W),    exp_keeper.average(b),
              exp_keeper.average(beta), exp_keeper.average(gamma),
              batch_mean,               batch_var,              ema]
    layer = cnn_lib.get_conv_layer_test_only(ema_model.inp,params)
    ema_model.add(Layer(layer,params))
    
    # SECOND CONV LAYER
    W,b,beta,gamma,batch_mean,batch_var,ema = model.layers[1].params
    exp_keeper_ops = exp_keeper_ops+[exp_keeper.apply([W, b, beta, gamma])]
    params = [exp_keeper.average(W),    exp_keeper.average(b),
              exp_keeper.average(beta), exp_keeper.average(gamma),
              batch_mean,               batch_var,              ema]
    layer = cnn_lib.get_conv_layer_test_only(ema_model.out,params)
    ema_model.add(Layer(layer,params))
    
    layer = cnn_lib.maxpool2d(ema_model.out,k=2)
    ema_model.add(Layer(layer,[]))
    
    # THIRD CONV LAYER
    W,b,beta,gamma,batch_mean,batch_var,ema = model.layers[3].params
    exp_keeper_ops = exp_keeper_ops+[exp_keeper.apply([W, b, beta, gamma])]
    params = [exp_keeper.average(W),    exp_keeper.average(b),
              exp_keeper.average(beta), exp_keeper.average(gamma),
              batch_mean,               batch_var,              ema]
    layer = cnn_lib.get_conv_layer_test_only(ema_model.out,params)
    ema_model.add(Layer(layer,params))
    
    # FORTH CONV LAYER
    W,b,beta,gamma,batch_mean,batch_var,ema = model.layers[4].params
    exp_keeper_ops = exp_keeper_ops+[exp_keeper.apply([W, b, beta, gamma])]
    params = [exp_keeper.average(W),    exp_keeper.average(b),
              exp_keeper.average(beta), exp_keeper.average(gamma),
              batch_mean,               batch_var,              ema]
    layer = cnn_lib.get_conv_layer_test_only(ema_model.out,params)
    ema_model.add(Layer(layer,params))
    
    layer = cnn_lib.maxpool2d(ema_model.out,k=2)
    ema_model.add(Layer(layer,[]))
    
    # FIFTH CONV LAYER
    W,b,beta,gamma,batch_mean,batch_var,ema = model.layers[6].params
    exp_keeper_ops = exp_keeper_ops+[exp_keeper.apply([W, b, beta, gamma])]
    params = [exp_keeper.average(W),    exp_keeper.average(b),
              exp_keeper.average(beta), exp_keeper.average(gamma),
              batch_mean,               batch_var,              ema]
    layer = cnn_lib.get_conv_layer_test_only(ema_model.out,params)
    ema_model.add(Layer(layer,params))
    
    # SIXTH CONV LAYER
    W,b,beta,gamma,batch_mean,batch_var,ema = model.layers[7].params
    exp_keeper_ops = exp_keeper_ops+[exp_keeper.apply([W, b, beta, gamma])]
    params = [exp_keeper.average(W),    exp_keeper.average(b),
              exp_keeper.average(beta), exp_keeper.average(gamma),
              batch_mean,               batch_var,              ema]
    layer = cnn_lib.get_conv_layer_test_only(ema_model.out,params)
    ema_model.add(Layer(layer,params))
    
    layer = cnn_lib.maxpool2d(ema_model.out,k=2)
    ema_model.add(Layer(layer,[]))
    
    layer = Layer(cnn_lib.flatten(ema_model.out),[])
    layer.flatten_dim = [6,6]
    ema_model.add(layer)    
    
    # FIRST DENSE LAYER
    W,b,beta,gamma,batch_mean,batch_var,ema = model.layers[10].params
    exp_keeper_ops = exp_keeper_ops+[exp_keeper.apply([W, b, beta, gamma])]
    params = [exp_keeper.average(W),    exp_keeper.average(b),
              exp_keeper.average(beta), exp_keeper.average(gamma),
              batch_mean,               batch_var,              ema]
    layer = cnn_lib.get_dense_layer_test_only_true(ema_model.out,   params,  True)
    ema_model.add(Layer(layer,params))
    
    layer = cnn_lib.dropout(layer, 1.0)
    ema_model.add(Layer(layer,[]))    

    # SECOND DENSE LAYER
    W,b,beta,gamma,batch_mean,batch_var,ema = model.layers[12].params
    exp_keeper_ops = exp_keeper_ops+[exp_keeper.apply([W, b, beta, gamma])]
    params = [exp_keeper.average(W),    exp_keeper.average(b),
              exp_keeper.average(beta), exp_keeper.average(gamma),
              batch_mean,               batch_var,              ema]
    layer = cnn_lib.get_dense_layer_test_only_true(ema_model.out,   params,  True)
    ema_model.add(Layer(layer,params))
    
    layer = cnn_lib.dropout(layer, 1.0)
    ema_model.add(Layer(layer,[]))    

    # FINAL LAYER
    W,b = model.layers[14].params    
    exp_keeper_ops = exp_keeper_ops+[exp_keeper.apply([W, b])]
    params = [exp_keeper.average(W),    exp_keeper.average(b)]
    layer = cnn_lib.get_dense_layer_test_only_true(ema_model.out,  params, False)
    ema_model.add(Layer(layer,params))

    maintain_averages_op = tf.group(*exp_keeper_ops)

    return ema_model,maintain_averages_op
    
    
    """


       
#model = Model([36,36])
#model.add_easy_layer(ltype="conv",filters_shape=[3,3],n_filters=32)
    
# TODO BETTER SAVER
"""        
def get_saver():  

    var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="global_stats")
    var.append(tf.trainable_variables())
    print("get_saver: I considered the following variables:")
    print(var)
    
    return tf.train.Saver(var)
""" 
    
"""
x = tf.placeholder(tf.float32,[None,128*9*9])
m,params = cnn_lib.get_dense_layer("",x,[128*9*9,256],None,False)
x_f = tf.placeholder(tf.float32,[1,9,9,128])
m_f = cnn_lib.get_dense_layer_test_only(x_f,params,[9,9],False)

img2 = np.ones((1,9,9,128))
img = np.reshape(img2,[1,9*9*128])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
a = sess.run(m,{x:img})
b = sess.run(m_f,{x_f:img2})
"""

"""
x = tf.placeholder(tf.float32,[None,36,36,1])
phase_train = tf.placeholder(tf.bool,[])
#phase_train = None
m=detector36(x,phase_train)

x_full = tf.placeholder(tf.float32,[1,500,500,1])
full_model = get_full_image_detector(m,x_full)

img = np.random.rand(1,500,500,1)
result1 = np.zeros((500,500))
result2 = np.zeros((500,500))

l=8

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    result2 = sess.run(full_model.layers[l].op,feed_dict={full_model.inp:img})
    #result2 = cnn_lib.mult_unstrided(result2,2)
        
    print("finished first")
    for i in range(18,500-18,1):
        print(i)
        for j in range(18,500-18,1):
            patch = img[:,i-18:i+18,j-18:j+18,:]
            patch = sess.run(m.layers[0].op,feed_dict={m.inp:patch,phase_train:False})
            break
   
    patch = img[:,0:36,0:36,:]
    patch = sess.run(m.layers[l+1].op,feed_dict={m.inp:patch,phase_train:False})
        
    
a2=result2[0,0,0,:]
#a = cnn_lib.unstrided(a)
b = np.squeeze(patch)
print(a2==b)
"""
 