# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 17:02:15 2017


spline test


@author: eduardo
"""


import scipy
import funcs.xml_reader as reader
from matplotlib import pyplot as plt
import numpy as np
mass = 36

path = "/media/eduardo/TOSHIBA EXT/INESC/INbreast/AllXML/20586908.xml"
points = reader.get_unprocessed_mask_points(path)
points = points[mass][0]

points2 = np.concatenate((points,points[0,np.newaxis,:]))
theta = 2 * np.pi * np.linspace(0, 1, points.shape[0]+1)


cs = scipy.interpolate.CubicSpline(theta, points2, bc_type='periodic')
xs = 2 * np.pi * np.linspace(0, 1, 3000)

#plt.scatter(points2[:,0],points2[:,1])
plt.plot(cs(xs)[:,0],cs(xs)[:,1],c="r")


