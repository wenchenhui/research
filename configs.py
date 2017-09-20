
import data.cbis_standart_format as cbis_standart_format

""" CBIS_STANDART_FORMAT """
scale = 1/24
clahe = False

src_location = "/media/eduardo/TOSHIBA EXT/"
folder_images = "raw CBIS mass/DOI/"
folder_masks = "raw CBIS mass rois/DOI/"
dst_location = "/media/eduardo/TOSHIBA EXT/tmp_testdataset/"

debug = True


cbis_standart_format.make_CBIS_standart_format(scale,clahe,src_location,folder_images,folder_masks,dst_location,debug)

""" Simple Data Processor """

halfpatch= 18

no_transformations = 5
use_rotations = True
use_mirroring = True
use_elastic_deform = True

src_location = "/media/username/TOSHIBA EXT/dataset2/"
dst_path = "/home/username/dataset_test"

debug = True