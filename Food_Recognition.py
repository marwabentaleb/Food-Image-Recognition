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

# Test the data augmentation 
test = x_train[0]
x = test.reshape((1,) + test.shape)
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='./preview', save_prefix='img', save_format='jpg'):
    i += 1
    if i > 20:
        break 

# Reshape the images    
target_size = 299
def reshape_data(X,target_size):
    X_reshaped = np.zeros((X.shape[0],target_size,target_size,3))
    for i in range(X.shape[0]):
        X_reshaped[i] = imresize(X[i], (target_size, target_size)) 
    return X_reshaped

# Applying data generation for test data et training data 
X_reshaped_train = reshape_data(x_train,target_size)
train_datagen = datagen.flow(X_reshaped_train, y_train_cat, batch_size=20, seed=11)

X_test_reshaped = reshape_data(x_test,target_size)
test_datagen = datagen.flow(X_test_reshaped, y_test_cat, batch_size=20, seed=11)

# Training
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (target_size, target_size, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, init='uniform', activation = 'relu'))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(units = 200, init='uniform', activation = 'relu'))
classifier.add(Dense(units = nb_class, activation = 'softmax'))

opt = SGD(lr=.01, momentum=.9)
classifier.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.fit_generator(train_datagen,
                         steps_per_epoch = x_train.shape[0],
                         epochs = 10,
                         validation_data = test_datagen,
                         validation_steps = x_test.shape[0])

# Model Evaluation

y_pred = classifier.predict(X_test_reshaped)
y_pred = y_pred[:,1]
y_pred = y_pred.astype(int)
y_test = y_test - 1

cnf_matrix = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=2)

class_names = [id_to_class[i] for i in range(nb_class)]


def plot_confusion_matrix(cm, classes,
                          normalize,
                          
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized confusion matrix"
        print("Normalized confusion matrix")
    else:
        title ='Confusion matrix, without normalization' 
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
# Plot confusion matrix
plt.figure()
fig = plt.gcf()
fig.set_size_inches(5, 5)
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, cmap=plt.cm.cool)
plt.show()

# Accuracy by class
import collections
corrects = collections.defaultdict(int)
incorrects = collections.defaultdict(int)
for (pred, actual) in zip(y_pred, y_test):
    if pred == actual:
        corrects[actual] += 1
    else:
        incorrects[actual] += 1

class_accuracies = {}
for ix in range(nb_class):
    class_accuracies[ix] = corrects[ix]/(corrects[ix] + incorrects[ix])
    
plt.grid(True)
plt.plot([0,1], list(class_accuracies.values()), 'o', color='black');
plt.title('Accuracy by Class')
plt.xlabel('label')
plt.ylabel('accuracy')

sorted_class_accuracies = sorted(class_accuracies.items(), key=lambda x: -x[1])
[(id_to_class[c[0]], c[1]) for c in sorted_class_accuracies]
for i in range(nb_class):
    print ("The accuracy of the class ",sorted_class_accuracies[i][0]," is ", round(sorted_class_accuracies[i][1],3))

# ..................................................... END ..................................................... #
