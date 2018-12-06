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
        # to add th class name for each image
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

# Create a new folder that contains a test folder and training folder
import pathlib
try :
    pathlib.Path('./splitted_data').mkdir(parents=True, exist_ok=True) 
except OSError:
    pass

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
from shutil import copyfile

for i in range(1,nb_class+1):
    class_dir = os.path.join(root_dir,str(i))
    all_files = os.listdir(class_dir)
    for nb_img in range((int) ((len(all_files)/100)*80)):
        copyfile(os.path.join(class_dir,all_files[nb_img]),
                 os.path.join('./splitted_data/training_set',str(i),all_files[nb_img])) 
        
    for nb_img in range((int) ((len(all_files)/100)*80),len(all_files)-1):
        copyfile(os.path.join(class_dir,all_files[nb_img]),
                 os.path.join('./splitted_data/test_set',str(i),all_files[nb_img])) 






















