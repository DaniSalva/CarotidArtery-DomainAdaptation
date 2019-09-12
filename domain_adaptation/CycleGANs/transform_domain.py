import keras
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import scipy
import numpy as  np

import cv2
import os
import matplotlib.pyplot as plt

path_images = "datasets/regicor_generalization/testregicor_raw/"
path_save = "datasets/regicor_generalization/testregicor_translated/"

import os
if not os.path.exists(path_save):
    os.makedirs(path_save)

def imread(path):
    return scipy.misc.imread(path).astype(np.float)
def load_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (448,448))
    img = img/127.5 - 1.
    return img[np.newaxis, :, :]
model = load_model('models_Gen/model_regA2regB.h5',custom_objects={'InstanceNormalization': InstanceNormalization()})

#loop through image_path_list to open each image
for imagePath in os.listdir(path_images):
    fullpath = os.path.join(path_images, imagePath)
    print (fullpath)
    image = load_img(fullpath)
    translated = model.predict(image)[0,:]
    new_image = os.path.join(path_save, '{}'.format(str(imagePath)))
    translated_img = 0.5 * translated + 0.5
    scipy.misc.imsave(new_image, translated_img)
