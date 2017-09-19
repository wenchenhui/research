import xml.etree.ElementTree as ET
import numpy as np
import os

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