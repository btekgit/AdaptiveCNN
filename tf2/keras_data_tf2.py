#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:26:25 2019

@author: btek
"""
from tensorflow.keras.datasets import mnist,fashion_mnist, cifar10,imdb
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow.keras.backend as K


def load_dataset(dset, normalize_data, options):
    if dset=='mnist':
        # input image dimensions
        img_rows, img_cols = 28, 28
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print(x_train.shape)
        n_channels=1

    elif dset=='cifar10':
        img_rows, img_cols = 32,32
        n_channels=3

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
    elif dset=='fashion':
        img_rows, img_cols = 28,28
        n_channels=1

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        
    elif dset=='mnist-clut':
        
        img_rows, img_cols = 60, 60  
        # the data, split between train and test sets
        
        #folder='/media/home/rdata/image/'
        folder='/home/btek/datasets/image/'
        data = np.load(folder+"mnist_cluttered_60x60_6distortions.npz", allow_pickle=True)
        y_trn =data['y_train']
        y_val = data['y_valid']
        y_tst = data['y_test']
        x_train, y_train = data['x_train'], np.argmax(y_trn,axis=-1)
        x_valid, y_valid = data['x_valid'], np.argmax(y_val,axis=-1)
        x_test, y_test = data['x_test'], np.argmax(y_tst,axis=-1)
        x_train=np.vstack((x_train,x_valid))
        y_train=np.concatenate((y_train, y_valid))
        n_channels=1
        normalize_data = False  # this dataset is already somehow normalized
        
        #decay_epochs =[e_i*30,e_i*100]
            
    elif dset=='lfw_faces':
        from sklearn.datasets import fetch_lfw_people
        lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=0.4)
        
        # introspect the images arrays to find the shapes (for plotting)
        n_samples, img_rows, img_cols = lfw_people.images.shape
        n_channels=1
        
        X = lfw_people.data
        n_features = X.shape[1]
        
        # the label to predict is the id of the person
        y = lfw_people.target
        target_names = lfw_people.target_names
        num_classes = target_names.shape[0]
        
        from sklearn.model_selection import train_test_split
        
        #X -= X.mean()
        #X /= X.std()
        #split into a training and testing set
        x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25)
       

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], n_channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], n_channels, img_rows, img_cols)
        input_shape = (n_channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, n_channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, n_channels)
        input_shape = (img_rows, img_cols, n_channels)
        ''' why the hell, I have written this?? BTEK
        if(n_channels==1):
            x_train = np.repeat(x_train,3, axis=3)
            x_test = np.repeat(x_test,3, axis=3)
            n_channels=3
            input_shape = (img_rows, img_cols, n_channels)
        '''
    num_classes = np.shape(np.unique(y_train))[0]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    if normalize_data:
        #Simple norm 0.1
        #x_train /= 255
        #x_test /= 255
        
        #Standard norm mean 0 , std 1, per input 
        #this normalization is very bad. BTEK for IMAGES
        #trn_mn = np.mean(x_train, axis=0) this normalization is very bad. BTEK for IMAGES
        #trn_std = np.std(x_train, axis=0) this normalization is very bad. BTEK for IMAGES
        
        # Standard for mean 127 and std per image.
        # This does not have 0 mean  but some negative value
        # Std is 1.0 used in may 2020 results
        # trn_mn = np.mean(x_train)
        # trn_std = np.std(x_train)
        # x_train -= 127.0   # I use this because other normalizations do not create symmetric distribution.
        # x_test -= 127.0
        # x_train/=(trn_std+1e-7)
        # x_test/=(trn_std+1e-7)
        # print("Data normed Mean(train):", np.mean(x_train), " Std(train):", np.std(x_train))
        # print("Data normed Mean(test):", np.mean(x_test), " Std(test):", np.std(x_test))
        
        x_train /= 255.0
        x_test /= 255.0
        trn_mn = np.mean(x_train)
        trn_std = np.std(x_train)
        x_train -= trn_mn   # I use this because other normalizations do not create symmetric distribution.
        x_test -= trn_mn
        #x_train/=(trn_std+1e-7)
        #x_test/=(trn_std+1e-7)
        print("Data normed Mean(train):", np.mean(x_train), " Std(train):", np.std(x_train))
        print("Data normed Mean(test):", np.mean(x_test), " Std(test):", np.std(x_test))
        
        # Standard for mean 127 and std per image.
        # This does not have 0 mean  and std is not 1.0
        # Std is 
#        x_train /= (255/4)
#        x_test /= (255/4)
#        x_train -= 2.0
#        x_test  -=  2.0
#        print("Data normed Mean(train):", np.mean(x_train), " Std(train):", np.std(x_train))
#        print("Data normed Mean(test):", np.mean(x_test), " Std(test):", np.std(x_test))
        
        # non-zero normalization.
#        trn_mn = np.mean(x_train[np.nonzero(x_train)])
#        trn_std = np.std(x_train[np.nonzero(x_train)])
#        x_train[np.nonzero(x_train)] -= trn_mn
#        x_test[np.nonzero(x_test)] -= trn_mn
#        print("Data normed Mean(train):", np.mean(x_train), " Std(train):", np.std(x_train))
#        print("Data normed Mean(test):", np.mean(x_test), " Std(test):", np.std(x_test))
#        x_train/=(trn_std+1e-7)
#        x_test/=(trn_std+1e-7)
#        print("Data normed Mean(train):", np.mean(x_train), " Std(train):", np.std(x_train))
#        print("Data normed Mean(test):", np.mean(x_test), " Std(test):", np.std(x_test))
        
        
        
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    return x_train,y_train,x_test,y_test,input_shape, num_classes


from tensorflow.keras.preprocessing import sequence
def load_textdataset(dset, settings, limit_train=None):
    if dset=='imdb':
        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=settings['top_words'])
        max_review_length = settings['max_review_length']
        X_train = sequence.pad_sequences(X_train,maxlen=max_review_length)
        X_test = sequence.pad_sequences(X_test,maxlen=max_review_length)
        input_shape =max_review_length
        num_classes = 2

    return X_train,y_train,X_test,y_test
    