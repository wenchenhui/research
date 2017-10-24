# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:48:22 2017

@author: eduardo
"""

import cnns.cnn_models as models
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


tf.reset_default_graph()
model = models.Model((50,50),1)
model.add_easy_layer(ltype = "conv_rotat",filters_shape = [5,5],n_filters = 16,padding="SAME",stride=1,activation="relu")
model.add_easy_layer(ltype = "conv_rotat_changed",filters_shape = [5,5],n_filters = 32,padding="SAME",stride=1,activation="relu")
model.add_easy_layer(ltype="max_pool",k=50, stride=50)
model.add_easy_layer(ltype="flatten")
model._compile()        

img = np.random.rand(1,50,50,1)
img2 = np.rot90(img, axes=(1,2))

sess = tf.Session()
sess.run(tf.global_variables_initializer())


res1 = sess.run(model.layers[0].out,feed_dict={model.inp:img})
res2 = sess.run(model.layers[0].out,feed_dict={model.inp:img2})


res3 = np.roll(res1,4,axis=3)
res3 = np.rot90(res3,axes=(1,2))

res2=np.squeeze(res2)
res3=np.squeeze(res3)

print("maximum difference = ",np.abs(res2-res3).max())
print("mean = ",np.mean(res3))
