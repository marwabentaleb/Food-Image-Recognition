#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 18:40:40 2018

@author: marwaa
"""

# Loading and Preprocessing Dataset

import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import os
from shutil import copyfile
from random import randint
from scipy.misc import imresize
import ast
from keras.utils.np_utils import to_categorical

# Path to data
root_dir = './UECFOOD256/'

rows = 2
cols = 3
fig, ax = plt.subplots(rows, cols,
                       frameon=False,
                       figsize=(7, 5))
fig.suptitle('Random Image from Some Food Class', fontsize=20)
sorted_food_dirs = sorted(os.listdir(root_dir))

for i in range(rows):
    for j in range(cols):
        try:
            # to choose the class of the food
            food_dir = sorted_food_dirs[i*cols + j]
            while (food_dir == 'bb_info.txt'):
                food_dir = sorted_food_dirs[i*cols + j]
        except:
            break
        # list of all the food images in the folder
        all_files = os.listdir(os.path.join(root_dir, food_dir))
        # Choosing a random image
        rand_img = np.random.choice(all_files)
        # transform the image to an array (M,N,3) "RGB"
        image = plt.imread(os.path.join(root_dir, food_dir, rand_img))
        # plot the image
        ax[i][j].imshow(image)
        # to add the class name for each image
        ec = (0, .6, .1)
        fc = (0, .7, .2)
        ax[i][j].text(0, -20, food_dir, size=10, rotation=0,
                ha="left", va="top", 
                bbox=dict(boxstyle="round", ec=ec, fc=fc))
plt.setp(ax, xticks=[], yticks=[])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

classes = []
with open('./UECFOOD256/category.txt', 'r') as txt:
    for l in txt.readlines():
        carac = ''
        i =0
        while((l.strip()[i] != '\t') and i < len(l.strip())-1):
            carac = carac + l.strip()[i]
            i+=1
        classes += [l.strip().replace(carac + '\t', '')]

del classes[0]

nb_class = len(classes)

id_to_class = dict(zip(range(len(classes)), classes))

# Number of training observations initialized to 0
nb_train = 0 
# Number of test observations initialized to 0
nb_test = 0

# Create the splitted data if haven't already existe and calculate nb_train and nb_test
if not os.path.isdir('./splitted_data') :
    import pathlib
    try :
        pathlib.Path('./splitted_data').mkdir(parents=True, exist_ok=True) 
    except OSError:
        pass
    
    # Create a new folder that contains a test folder and training folder 
    try :
        pathlib.Path('./splitted_data/training_set').mkdir(parents=True, exist_ok=True) 
        pathlib.Path('./splitted_data/test_set').mkdir(parents=True, exist_ok=True) 
    except OSError:
        pass
    for i in range(1,nb_class+1):
        try :
            pathlib.Path(os.path.join('./splitted_data/training_set/',str(i))).mkdir(parents=True, exist_ok=True) 
            pathlib.Path(os.path.join('./splitted_data/test_set/',str(i))).mkdir(parents=True, exist_ok=True) 
        except OSError:
            pass

    # Split the data into test set ans training set
    for i in range(1,nb_class+1):
        class_dir = os.path.join(root_dir,str(i))
        all_files = os.listdir(class_dir)
        nb_train = nb_train + (int) ((len(all_files)/100)*80)
        nb_test = nb_test + (len(all_files)- (int) ((len(all_files)/100)*80))
        for nb_img in range((int) ((len(all_files)/100)*80)):
            copyfile(os.path.join(class_dir,all_files[nb_img]),
                     os.path.join('./splitted_data/training_set',str(i),all_files[nb_img])) 
            
        for nb_img in range((int) ((len(all_files)/100)*80),len(all_files)-1):
            copyfile(os.path.join(class_dir,all_files[nb_img]),
                     os.path.join('./splitted_data/test_set',str(i),all_files[nb_img])) 
else:
    print('Train/Test files already copied into separate folders.')



#  Mixed data 
def mix_data(X, Y):
    X_result = []
    Y_result = []
    nb = len(X)
    for i in range(nb):
        pos = randint(0,nb - i - 1)
        X_result.append(X[pos])
        Y_result.append(Y[pos])
        del X[pos]
        del Y[pos]
    return X_result, Y_result

# Function to load the images after resizing it
def load_images(root, min_side=299):
    all_imgs = []
    all_classes = []
    resize_count = 0
    invalid_count = 0
    for i, subdir in enumerate(os.listdir(root)):
        imgs = os.listdir(os.path.join(root, subdir))
        for img_name in imgs:
            if img_name != 'bb_info.txt' :
                img_arr = img.imread(os.path.join(root, subdir, img_name))
                img_arr_rs = img_arr
                w, h, _ = img_arr.shape
                try:
                    w, h, _ = img_arr.shape
                    if w < min_side:
                        wpercent = (min_side/float(w))
                        hsize = int((float(h)*float(wpercent)))
                        print('new dims:', min_side, hsize)
                        img_arr_rs = imresize(img_arr, (min_side, hsize))
                        resize_count += 1
                    elif h < min_side:
                        hpercent = (min_side/float(h))
                        wsize = int((float(w)*float(hpercent)))
                        print('new dims:', wsize, min_side)
    
                        img_arr_rs = imresize(img_arr, (wsize, min_side))
                        resize_count += 1
                    all_imgs.append(img_arr_rs)
                    all_classes.append(ast.literal_eval(subdir))
                except:
                    print('Skipping bad image: ', subdir, img_name)
                    invalid_count += 1
    print(len(all_imgs), 'images loaded')
    print(resize_count, 'images resized')
    print(invalid_count, 'images skipped')
    all_imgs_mixed, all_classes_mixed =  mix_data(all_imgs,all_classes)
    return np.array(all_imgs_mixed), np.array(all_classes_mixed), resize_count, invalid_count

# Load the training set after resizing 
root = './splitted_data/training_set'
x_train, y_train, resize_count, invalid_count = load_images(root , min_side=299)

# Load the test set after resizing 
root = './splitted_data/test_set'
x_test, y_test, resize_count, invalid_count = load_images(root , min_side=299)


# Function to plot Food images
def show_pics(rows, cols, X, Y):
    fig, ax = plt.subplots(rows, cols,
                           frameon=False,
                           figsize=(7, 5))
    fig.suptitle('Food Image', fontsize=20)
    for i in range(rows):
        for j in range(cols):
            pos = randint(0,len(X)-1)
            ax[i][j].imshow(X[pos])
            # to add the class name for each image
            ec = (0, .6, .1)
            fc = (0, .7, .2)
            ax[i][j].text(0, -20, Y[pos], size=10, rotation=0,
                          ha="left", va="top", 
                          bbox=dict(boxstyle="round", ec=ec, fc=fc))
            plt.setp(ax, xticks=[], yticks=[])

show_pics(rows, cols, x_train, y_train)
show_pics(rows, cols, x_test, y_test)

# Converts a class vector (integers) to binary class matrix 
# Remove the first column because the labels starts from 1 
# Or "to_categorical" suppose that labels starts from 0
y_train_cat = to_categorical(y_train)[:,1:]
y_test_cat = to_categorical(y_test)[:,1:]

# Image Augmentation
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# test datagen
x = x_train[0].reshape((1,) + x_train[0].shape)
i = 0
for batch in datagen.flow(x_train, batch_size=len(),
                          save_to_dir='./preview', save_prefix='cat', 				  save_format='jpg'):
    i += 1
    if i > 20:
        break 
    print(batch)
batch_size=79
#nb_batch = (int) (len(x_train)/batch_size)
#x_train_shaped = x_train.reshape((nb_batch,)+ x_train[0].shape)
#train_datagen = datagen.flow(x_train, y_train_cat, batch_size=batch_size, seed=11)
