
import glob
import dicom
import pandas as pd
from matplotlib import pyplot as plt
import xml_reader
import numpy as np
import os
import scipy
import cv2

path = "/home/eduardo/tese/data/INbreast"


def get_images_path():
    paths = glob.glob(path+"/AllDICOMs/*.dcm")
    paths.sort()
    return paths
    
def separate_sets():
    images = get_images_path()
    with_masses = set()
    no_masses = set()
    
    for img in images:
        pat = os.path.basename(img).split("_")[0]
        if len(get_masses(pat))==0:
            no_masses.add(img)
        else:
            with_masses.add(img)
            
    return with_masses,no_masses
    
def get_xml_path():
    paths = glob.glob(path+"/AllXML/*.xml")
    paths.sort()    
    return paths
    
def read_img(img_path):
    return dicom.read_file(img_path).pixel_array
    
def show_img(img):
    plt.imshow(img,cmap = "gray")
   
def read_info():
    df = pd.read_excel(path+"/INbreast.xls")
    return df
    
def has_mass(df,patient):
    return df.loc[df["File Name"] == patient].iloc[0]["Mass"]=="X"
     
clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))

def preprocessing(img,resize_factor=(1/24)):
    img = cv2.resize(img, (0,0), fx=resize_factor,fy=resize_factor)
    cl1 = clahe.apply(img)
    cl1 = (cl1-np.min(cl1))/(np.max(cl1)-np.min(cl1))-0.5
    return cl1
    
def get_masses(pat,resize_factor=(1/24)):
    all_xmls = get_xml_path()
    masses = []
    xml = path+"/AllXML/"+pat+".xml"
    if xml in all_xmls:
            lesions = xml_reader.get_mask_points(xml)
            for row in (lesions):
                if row[2] == "M":
                    masses.append([np.round(row[0]*resize_factor).astype(int),
                                   np.round(row[1]*resize_factor).astype(int),row[2]])
    return masses
    
    
    
def make_histogram(list_of_values):
    bins = np.linspace(min(list_of_values)-1,max(list_of_values)+1,10)
    plt.xlim([min(list_of_values)-5, max(list_of_values)+5])
    plt.hist(list_of_values, bins=bins, alpha=0.5)
    plt.title("Mass sizes")
    plt.xlabel('max bbox side (10 evenly spaced bins)')
    plt.ylabel('count')
    plt.show()
    
    values = np.asarray(list_of_values)
    values = np.sort(values)
    
    frequency = np.arange(values.shape[0])/values.shape[0]
    plt.plot(values,frequency,"b")
    plt.title("Mass frequency vs mass maximum size")
    plt.xlabel("max size")
    plt.ylabel("frequency")
    plt.show()
    
def check_sizes():
    masses = [] 
    all_xmls = get_xml_path()
    for image in get_images_path():
        pat = os.path.basename(image).split("_")[0]
        xml = path+"/AllXML/"+pat+".xml"
               
        if xml in all_xmls:
            lesions = xml_reader.get_mask_points(xml)
            
            for row in (lesions):
                if row[2] == "M":
                    masses.append(row)
        
    size = []
    for mass in masses:
        size.append(max(mass[1][2]-mass[1][0],mass[1][3]-mass[1][1]))
        
    make_histogram(size)

    