import numpy as np
import glob
import os
import cv2
import scipy.misc

path_GT_SS = "C:/Users/dsalvadu/Documents/Master/TFM/Environment_Semantic_Segmentation/SemSeg/Datasets_SemSeg/dataset_regicor2nefrona/data/test18_gt/"
path_GT_BS = "C:/Users/dsalvadu/Documents/Master/TFM/Environment_Semantic_Segmentation/SemSeg/Datasets_SemSeg/dataset_regicor2nefrona/data/test18_gt_2labels/"

if not os.path.exists(path_GT_BS):
    os.makedirs(path_GT_BS)

def load_img(path):
    img_data = cv2.imread(path)
    return img_data

# loop through image_path_list to open each image
for imagePath in os.listdir(path_GT_SS):
    fullpath = os.path.join(path_GT_SS, imagePath)
    lab_data = load_img(fullpath)
    new_image = os.path.join(path_GT_BS, '{}'.format(str(imagePath)))

    #cv2.imshow('img_pre',lab_data)

    lab_rename = lab_data
    '''
        Rename labels

        near_wall --> 0
        lumen_bulb --> 0
        lumen_cca --> 0
        imt_bulb --> 0
        imt_cca --> 1
        far_wall --> 0
    '''
    c0_idx = np.where(lab_data == 0)
    c1_idx = np.where(lab_data == 1)
    c2_idx = np.where(lab_data == 2)
    c3_idx = np.where(lab_data == 3)
    c4_idx = np.where(lab_data == 4)
    c5_idx = np.where(lab_data == 5)

    lab_rename[c0_idx] = 0
    lab_rename[c1_idx] = 0
    lab_rename[c2_idx] = 0
    lab_rename[c3_idx] = 0
    lab_rename[c4_idx] = 1
    lab_rename[c5_idx] = 0

    #cv2.imshow('img',(lab_data*255).astype('uint8'))
    #cv2.waitKey(0)

    cv2.imwrite(new_image, lab_rename)
