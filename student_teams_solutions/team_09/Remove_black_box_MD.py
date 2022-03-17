#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Importing libraries

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from IPython.display import IFrame
#from io import BytesIO
from numpy import random
from PIL import Image, ImageFile
from scipy import signal
from scipy import ndimage
from scipy.ndimage import filters
from skimage import data
import cv2
from tqdm.notebook import tqdm


# In[7]:


# load datasets
path_annotrain = "../datasets/datasets_train/train_annotation/_annotation.csv"
path_footprint = "../datasets/car_models_footprint.csv"

train_annotation = pd.read_csv(path_annotrain, index_col=0)
footprints = pd.read_csv(path_footprint, index_col=0,delimiter=";")


# In[8]:


# split data to car and non car
train_annotation['class'] = np.where(train_annotation['class'] == 'car', "car", "non car")


# In[15]:


train_annotation = train_annotation[train_annotation['class'] == 'car']


# In[16]:


DIR="../datasets/datasets_train/train/"
images_path= train_annotation["im_name"].values
images_labels = train_annotation["models"].values
images_train = []
images_train_labels = []
for i in range(len(images_path)):
    try:
        image = Image.open(DIR+images_path[i])
        image = image.resize((416, 416))
        image = np.asarray(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        images_train.append(image)
        images_train_labels.append(images_labels[i])
    except:
        continue


# In[17]:


len(images_train)


# In[31]:


rows = 20
_ = plt.figure(figsize=(12, 12))
cols = 100//rows if 100 % 2 == 0 else 100//rows + 1
for i in range(100):
    plt.subplot(rows, cols, i+1)
    plt.imshow(images_train[-i],cmap="gray")
    plt.title(str(1309-i))
    plt.xticks([])
    plt.yticks([])


# In[89]:


im = images_train[1224]
plt.imshow(im)
#neigbor_denoising(ar, neig=5)


# In[126]:


gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
a= 1*(gray ==0)
a=a.astype('uint8')
kernel = np.ones((10,10),np.uint8)
dilation = cv2.dilate(a,kernel,iterations = 1)
#opening = cv2.morphologyEx(a, cv2.MORPH_OPEN, kernel)
#closing = cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel)
plt.imshow(dilation,cmap='gray')


# In[127]:


mask = dilation
dst = cv2.inpaint(im,mask,20,cv2.INPAINT_TELEA)
plt.imshow(dst)


# In[128]:


median = cv2.medianBlur(dst,5)
plt.imshow(median)


# In[6]:


def neigbor_denoising(ar, neig=5):
    """
    replace null pixels by averaging neighborhood not null pixels

    """
    data = ar.copy()
    mark_x = np.where(data == 0)[0]  # row index of null pixels
    mark_y = np.where(data == 0)[1]  # col index of null pixels
    for x, y in zip(mark_x, mark_y):
        # assuming you want neig x neig square
        slice = data[max(0, x-neig):x+neig, max(0, y-neig):y+neig]
        data[x, y] = np.mean([i for i in slice.flatten() if i > 0])
    return data


# In[103]:


denoise = neigbor_denoising(im, neig=5)
plt.imshow(denoise)

