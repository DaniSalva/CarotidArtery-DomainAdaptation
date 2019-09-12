import scipy
import scipy.misc
import numpy as np
import os
import cv2

path_images='C:/Users/dsalvadu/Documents/Master/TFM/data/results/Generalization_US/predict_nefrona/'
path_save='C:/Users/dsalvadu/Documents/Master/TFM/data/results/Generalization_US/predict_nefrona_resize/'

def resize_img(path):
    oriimg = cv2.imread(path)
    newimg = cv2.resize(oriimg, (445,470))
    return newimg

#loop through image_path_list to open each image
for imagePath in os.listdir(path_images):
    fullpath = os.path.join(path_images, imagePath)
    image_gt = resize_img(fullpath)
    new_image = os.path.join(path_save, '{}'.format(str(imagePath)))
    cv2.imwrite(new_image, image_gt)
