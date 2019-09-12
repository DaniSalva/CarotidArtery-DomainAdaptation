import scipy
import scipy.misc
import numpy as np
import os
import cv2

path_images='C:/Users/dsalvadu/Documents/Master/TFM/data/NEFRONA/ORIGINALS_cropped/'
path_save='C:/Users/dsalvadu/Documents/Master/TFM/data/NEFRONA/ORIGINALS_resized/'

if not os.path.exists(path_save):
    os.makedirs(path_save)

def resize_img(path):
    oriimg = cv2.imread(path)
    print(oriimg.shape)
    newimg = cv2.resize(oriimg, (445,470))
    return newimg

#loop through image_path_list to open each image
for imagePath in os.listdir(path_images):
    if imagePath.endswith('.png'):
        fullpath = os.path.join(path_images, imagePath)
        print(fullpath)
        image_gt = resize_img(fullpath)
        new_image = os.path.join(path_save, '{}'.format(str(imagePath)))
        cv2.imwrite(new_image, image_gt)
