
import data.cbis_standart_format as cbis_standart_format
import simple_data_processor

"""
    WHOLE PROCESS OF LEARNING A PATCH MODEL FROM FULL IMAGES
"""



""" CBIS_STANDART_FORMAT """
scale = 1/24
clahe = False

src_location = "/media/eduardo/TOSHIBA EXT/"
folder_images = "raw CBIS mass/DOI/"
folder_masks = "raw CBIS mass rois/DOI/"
dst_location = "/media/eduardo/TOSHIBA EXT/tmp_testdataset/"

debug = False


#cbis_standart_format.make_CBIS_standart_format(scale,clahe,src_location,folder_images,folder_masks,dst_location,debug)

""" Simple Data Processor """


src_location = dst_location
dst_location = "/home/username/tmp_testdataset2ndpart"

set_masses_size = {"train":491,"validation":100,"test":100}
set_neg_size = {"train":0,"validation":0,"test":0}

dataset = simple_data_processor.make_dataset(set_masses_size, set_neg_size, src_location, dst_location)

"""
OPOPOPOPOPOPOPOP
_______________

WARNING: YOU NEED TO SAVE THE DATASET. NOT DONE YET
_________________
OPOPOPOPOPOPOPOP
"""
halfpatch= 18

no_transformations = 5
use_rotations = True
use_mirroring = True
use_elastic_deform = True

debug = True
dataset.make_patches_dataset(no_transformations,use_rotations,use_mirroring,use_elastic_deform,debug)