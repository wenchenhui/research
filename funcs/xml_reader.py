import xml.etree.ElementTree as ET
import numpy as np
import os
import scipy
from scipy.ndimage.morphology import binary_fill_holes


def get_unprocessed_mask_points(path):
    if(not os.path.isfile(path)):
        return []
    tree = ET.parse(path)
    root = tree.getroot()
    
    numberOfROIs = root[0][1][0][3].text
    
    lesions = []
    inner = 0
    for lesion in root[0][1][0][5]:
        name = lesion[15].text
        Npoints = int(lesion[17].text)
        points = np.zeros((Npoints,2))
        
        for i in range(Npoints):
            text = lesion[21][i].text
            numT = text.strip("()").split(",")
            points[i,0] = float(numT[0])
            points[i,1] = float(numT[1])
         
        if name == "Calcification":
            lesions.append([points,"C"])
        elif name == "Mass":
            lesions.append([points,"M"])
        elif name == "Cluster":
            lesions.append([points,"R"])
        elif name == "Point1":
            print("ONE POINT ONE IGNORED")
        elif name == "Distortion": 
            lesions.append(([points,"R"]))
    return lesions
            
def get_mask_images(file,shape):

    lesions = get_unprocessed_mask_points(file)       
    masses = []
    for lesion in lesions:
        if lesion[1] == "M":
            masses.append(lesion[0])

    masses_imgs = []    
    for mass in masses:
        img_mass = np.zeros(shape,np.bool)
        points = np.concatenate((mass,mass[0,np.newaxis,:]))
        theta = 2 * np.pi * np.linspace(0, 1, mass.shape[0]+1)

        cs = scipy.interpolate.CubicSpline(theta, points, bc_type='periodic')
        xs = 2 * np.pi * np.linspace(0, 1, 30000)
        indexes = np.round(cs(xs)).astype(int)
        y,x = np.clip(indexes[:,0],0,shape[1]-1),np.clip(indexes[:,1],0,shape[0]-1)
        img_mass[x,y] = True
        masses_imgs.append(binary_fill_holes(img_mass))
    
    return masses_imgs
        
def has_mass(file):
    lesions = get_unprocessed_mask_points(file)       
    masses = []
    for lesion in lesions:
        if lesion[1] == "M":
            masses.append(lesion[0])
    return len(masses)
        
        
        
            
def get_mask_points(path):

    if(not os.path.isfile(path)):
        return []
    tree = ET.parse(path)
    root = tree.getroot()
    
    numberOfROIs = root[0][1][0][3].text
    
    lesions = []
    inner = 0
    for lesion in root[0][1][0][5]:
        name = lesion[15].text
        Npoints = int(lesion[17].text)
        points = np.zeros((Npoints,2))
        
        for i in range(Npoints):
            text = lesion[21][i].text
            numT = text.strip("()").split(",")
            points[i,0] = float(numT[0])
            points[i,1] = float(numT[1])
 
        pos,bbox=get_coordinates(points)
        
        if name == "Calcification":
            lesions.append([pos,bbox,"C"])
        elif name == "Mass":
            lesions.append([pos,bbox,"M"])
        elif name == "Cluster":
            lesions.append([pos,bbox,"R"])
        elif name == "Point1":
            print("ONE POINT ONE IGNORED")
        elif name == "Distortion": 
            lesions.append(([pos,bbox,"R"]))
    #print("Debug: xmlReader:",numberOfROIs,"regions",len(calc)+len(mass)+len(cluster),"documented")

    return lesions


def get_coordinates(lesion):
    xmin = np.min(lesion[:,0])
    xmax = np.max(lesion[:,0])
    ymin = np.min(lesion[:,1])
    ymax = np.max(lesion[:,1])
    
    pos = np.round(np.asarray([np.average([xmin,xmax]),np.average([ymin,ymax])])).astype(int)
    bbox = np.round(np.asarray([np.floor(xmin),np.floor(ymin),np.ceil(xmax),np.ceil(ymax)])).astype(int)
    #print("Debug: experimentINbreast: coordinates ",xmin,xmax,ymin,ymax,pos,bbox)
    return pos,bbox