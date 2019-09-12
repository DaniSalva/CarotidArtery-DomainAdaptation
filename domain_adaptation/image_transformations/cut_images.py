import cv2
import os
import matplotlib.pyplot as plt

#img = cv2.imread("datasets/nefrona2regicor/testA/2681-AC_D_0.png")
path_images = "datasets/nefrona2regicor/trainA/"
path_save = path_images + "cropped/"
#crop_img = img[46:617,133:744]
#cv2.imshow("img crop",crop_img)
#cv2.waitKey(0)

import os
if not os.path.exists(path_save):
    os.makedirs(path_save)

#loop through image_path_list to open each image
for imagePath in os.listdir(path_images):
    fullpath = os.path.join(path_images, imagePath)

    image = cv2.imread(fullpath)
    # display the image on screen with imshow()
    # after checking that it loaded
    if image is not None:
        print(fullpath)
    elif image is None:
        print ("Error loading: " + imagePath)
        continue
    # Please check if the size of original image is larger than the pixels you are trying to crop.
    (height, width, channels) = image.shape
    if height >= 400 and width >= 400:
        plt.imshow(image)
        plt.title('Original')
        plt.show()
        crop_img = image [46:617,133:744]

        # Problem lies with this path. Ambiguous like pic/home/ubuntu/Downloads/10_page.png.jpg
        print('pic{:>05}.jpg'.format(imagePath))

        # Try a proper path. A dirname + file_name.extension as below, you will have no problem saving.
        new_image = os.path.join(path_save,'{}'.format(str(imagePath)))
        print(new_image)
        cv2.imwrite(new_image, crop_img)
        plt.imshow(crop_img)
        plt.title('Cropped')
        plt.show()