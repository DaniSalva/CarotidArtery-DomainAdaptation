import os
import numpy as np
import shutil

# # Creating Train / Val / Test folders (One time use)
root_dir = 'datasets/regicor_generalization/TrainRegicor/'

os.makedirs('datasets/regicor_generalization/trainA')
os.makedirs('datasets/regicor_generalization/trainB')
os.makedirs('datasets/regicor_generalization/testA')
os.makedirs('datasets/regicor_generalization/testB')


allFileNames = os.listdir(root_dir)
np.random.shuffle(allFileNames)
trainA_FileNames, trainB_FileNames, testA_FileNames,testB_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)*0.45), int(len(allFileNames)*0.90),int(len(allFileNames)*0.95)])

trainA_FileNames = [root_dir+'/'+ name for name in trainA_FileNames.tolist()]
trainB_FileNames = [root_dir+'/' + name for name in trainB_FileNames.tolist()]
testA_FileNames = [root_dir+'/' + name for name in testA_FileNames.tolist()]
testB_FileNames = [root_dir+'/' + name for name in testB_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('TrainA: ', len(trainA_FileNames))
print('TrainB: ', len(trainB_FileNames))
print('TestingA: ', len(testA_FileNames))
print('TestingB: ', len(testB_FileNames))

# Copy-pasting images
for name in trainA_FileNames:
    shutil.copy(name, 'datasets/regicor_generalization/trainA')

for name in trainB_FileNames:
    shutil.copy(name, 'datasets/regicor_generalization/trainB')

for name in testA_FileNames:
    shutil.copy(name, 'datasets/regicor_generalization/testA')

for name in testB_FileNames:
    shutil.copy(name, 'datasets/regicor_generalization/testB')