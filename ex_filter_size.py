#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 15:50:29 2019

@author: btek
"""


from skimage import io, filters
from scipy import ndimage,misc
import numpy as np
import matplotlib.pyplot as plt


from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#inputimg = io.imread('turkish_flag_640.png')
inputimg = x_train[0]/255
#inputimg =inputimg[160:266,300:420,1]
#inputimg = ndimage.zoom(inputimg,0.5)

sh = (inputimg.shape[0],inputimg.shape[1],1)
outputimages = np.zeros(shape=[inputimg.shape[0],inputimg.shape[1],3],dtype='float32')
outputimages[:,:,0] = filters.gaussian(inputimg,sigma=1)
outputimages[:,:,1] = filters.sobel_h(inputimg)
outputimages[:,:,2] = filters.sobel_v(filters.gaussian(inputimg,sigma=2.5))
k3 = np.array([[-1,2,-1],]).repeat(3,axis=0).T
k5 = np.array([[-1,-1,4,-1,-1],]).repeat(5,axis=0).T
k9 = np.array([[-1,-1,-1,-1,8,-1,-1,-1,-1],]).repeat(9,axis=0).T
out_3 = ndimage.correlate(inputimg, k3,mode='constant',cval=0)
out_5 = ndimage.correlate(inputimg, k5,mode='constant',cval=0)
out_9 = ndimage.correlate(inputimg, k9,mode='constant',cval=0)
plt.figure()
plt.imshow(inputimg,cmap='gray')
plt.axis('off')
plt.figure()
plt.imshow(outputimages[:,:,1],cmap='gray')
plt.axis('off')
plt.figure()
plt.imshow(outputimages[:,:,2],cmap='gray')
plt.axis('off')
plt.figure()
plt.imshow(out_3,cmap='gray')
plt.axis('off')
plt.figure()
plt.imshow(out_5,cmap='gray')
plt.axis('off')
plt.figure()
plt.imshow(out_9,cmap='gray')
plt.axis('off')
