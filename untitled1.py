# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 11:00:15 2017

@author: eduardo
"""

import read_inbreast as readin
import os


paths = readin.get_images_path()
sizes = []
readin.resize_factor = 1/12

for path in paths:

    pat = os.path.basename(path).split("_")[0]
    masses = readin.get_masses(pat)    
    
    for mass in masses:
        bb = mass[1]
        biggest_size = max(bb[3]-bb[1],bb[2]-bb[0])
        sizes.append(biggest_size)
    
readin.make_histogram(sizes)