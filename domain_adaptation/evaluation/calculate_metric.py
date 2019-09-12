import os
import cv2
import numpy as np

path_images='C:/Users/dsalvadu/Documents/Master/TFM/data/results/Postprocessed_Matlab/IMG_postprocessed_exp4_nefrona/'
#path_images='C:/Users/dsalvadu/Documents/Master/TFM/data/results/ExpDA_2labels/test_epoch_12_Exp3_2labels/'
path_GT= 'C:/Users/dsalvadu/Documents/Master/TFM/data/NEFRONA/GT_cropped_448/'

dir_imgs = os.listdir(path_images)
dir_gt = os.listdir(path_GT)


def get_eval(target, predict):
    k=255
    TP = np.sum(np.logical_and(predict == k, target == k))
    TN = np.sum(np.logical_and(predict == 0, target == 0))
    FP = np.sum(np.logical_and(predict == k, target == 0))
    FN = np.sum(np.logical_and(predict == 0, target == k))
    return(TP, FP, TN, FN)

def calculate_iou(target,prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    return np.sum(intersection) / np.sum(union)

def calculate_iou_2(TP,FP,FN):
    return TP / (TP+FP+FN)

def calculate_dice(target,prediction):
    k = 255
    dice = np.sum(prediction[target == k]) * 2.0 / (np.sum(prediction) + np.sum(target))
    return dice

def calculate_pixel_accuracy(TP, FP, TN, FN):
    if TP==0 and TN==0:
        pixel_accuracy=0
    else:
        pixel_accuracy =(TP + TN) / (TP + TN + FP + FN)
    return pixel_accuracy

def calculate_specifity(FP, TN):
    if TN ==0:
        specificity = 0
    else:
        specificity =(TN) / (TN + FP)
    return specificity

def calculate_sensitivity(TP, FN):
    if TP ==0:
        sensitivity=0
    else:
        sensitivity =(TP ) / (TP + FN)
    return sensitivity

def calculate_precision(TP, FP):
    if TP ==0:
        prec = 0
    else:
        prec =(TP) / (TP + FP)
    return prec

def convert_whitepixels(image):
    white = [255, 255, 255]
    white_pre = [1, 1, 1]
    height, width, channels = image.shape

    for x in range(0, width):
        for y in range(0, height):
            channels_xy = image[y, x]
            if all(channels_xy == white_pre):
                image[y, x] = white

iou_scores = []
iou_2_scores = []
dice_scores = []
acc_scores = []
prec_scores = []
sen_scores = []
spec_scores = []

fullpath_img = os.path.join(path_images, dir_imgs[0])
fullpath_gt = os.path.join(path_GT, dir_gt[0])
image = cv2.imread(fullpath_img)
gt = cv2.imread(fullpath_gt)
print("Predictions:",image.shape," -- ",np.unique(image))
print("GT:",gt.shape," -- ",np.unique(gt))


for k in range(len(dir_imgs)):
    fullpath_img = os.path.join(path_images, dir_imgs[k])
    fullpath_gt = os.path.join(path_GT, dir_gt[k])

    image = cv2.imread(fullpath_img)
    gt = cv2.imread(fullpath_gt)
    height, width, channels = image.shape
    convert_whitepixels(image)

    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, gt) = cv2.threshold(gt, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    TP, FP, TN, FN = get_eval(gt, image)
    iou = calculate_iou(gt,image)
    iou2 = calculate_iou_2(TP,FP,FN)
    dice = calculate_dice(gt,image)
    acc = calculate_pixel_accuracy(TP, FP, TN, FN)
    prec = calculate_precision(TP,FP)
    sen = calculate_sensitivity(TP,FN)
    spec = calculate_specifity(FP,TN)

    iou_scores.append(iou)
    iou_2_scores.append(iou2)
    dice_scores.append(dice)
    acc_scores.append(acc)
    prec_scores.append(prec)
    sen_scores.append(sen)
    spec_scores.append(spec)

    '''cv2.imshow("pred",image)
    cv2.imshow("gt",gt)
    cv2.waitKey(0)
'''
print("Mean IoU: ", sum(iou_scores)/len(iou_scores))
print("Mean IoU2: ", sum(iou_2_scores)/len(iou_2_scores))
print("Mean Dice: ", sum(dice_scores)/len(dice_scores))
print("Mean Accuracy: ", sum(acc_scores)/len(acc_scores))
print("Mean Precision: ", sum(prec_scores)/len(prec_scores))
print("Mean Sensitivity: ", sum(sen_scores)/len(sen_scores))
print("Mean Specificity: ", sum(spec_scores)/len(spec_scores))

print(sum(iou_scores)/len(iou_scores),sum(dice_scores)/len(dice_scores),
      sum(acc_scores)/len(acc_scores), sum(prec_scores)/len(prec_scores),
        sum(sen_scores)/len(sen_scores), sum(spec_scores)/len(spec_scores))