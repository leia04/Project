import pandas as pd
import numpy as np
import cv2
import sys,os
sys.path.append(os.pardir)
import matplotlib.gridspec as gridspec
#from common.layers import *
from glob import glob
from tqdm import tqdm

# Path of the data
data_path = '/home/s2020102663/deep_sourcecode/dataset_pneumonia/chest_xray/chest_xray/'

# Set the path of each data
train_path = data_path + 'train/'
valid_path = data_path + 'val/'
test_path = data_path + 'test/'

print(f'train data : {len(glob(train_path + "*/*"))}')
print(f'validation data : {len(glob(valid_path + "*/*"))}')
print(f'test data : {len(glob(test_path + "*/*"))}')
print(test_path)
#print(glob(test_path + '*/*')[:6]) 이 코드가 왜 필요해?

def load_data(data_path):
    image_files = glob(data_path + "*/*")
    #print(image_files)
    images = []
    labels = []
###이미지가 아닌 요상한 파일이 None을 만들어서 resizer가 안되지jpeg만 뽑을 수 있게 만들자.
    for image_file in tqdm(image_files):
        if image_file.endswith('.jpeg') or image_file.endswith('.png') or image_file.endswith('.jpg'):
            label = 1 if 'PNEUMONIA' in image_file else 0

            img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            #print(img)
            img = cv2.resize(img, (28, 28))
            img = img.reshape(1, 1, 28, 28)
            #img = np.array(img)
            images.append(img/255.0)
            #print(images)
            labels.append(label)
        #print(images)
        #print(labels)
    return np.vstack(images), np.array(labels)

#if __name__ == '__main__':

# Load data from the each path

x_train, t_train = load_data(train_path)
x_valid, t_valid = load_data(valid_path)
x_test, t_test = load_data(test_path)
print(x_valid)
print(t_valid)
    #print("Shape of the x_train:", x_train.shape)
    #print("Shape of the x_valid:", x_valid.shape)
    #print("Shape of the x_test:", x_test.shape):w

def shuffle_dataset(x, t):
    mix = np.random.permutation(x.shape[0])
    x = x[mix]
    t = t[mix]

    return x, t

x_train_f, t_train_f = shuffle_dataset(x_train, t_train)
x_valid_f, t_valid_f = shuffle_dataset(x_valid, t_valid)
x_test_f, t_test_f = shuffle_dataset(x_test, t_test)

print('Shape of the x_train:',x_train.shape)
print('Shape of the x_valid:',x_valid.shape)
print('Shape of the x_test:',x_test.shape)

print('Number of Normal cases:',np.sum(t_train ==0))
print('Number of Pneumonia cases:',np.sum(t_train==1))






































