#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:57:27 2019

@author: btek
"""

from __future__ import print_function
import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
    os.environ['CUDA_VISIBLE_DEVICES']="0"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from keras_data_tf2 import load_dataset
#from keras.datasets import cifar10
import numpy as np

from keras_utils_tf2 import  ClipCallback
from resnet_s_tf2 import resnet






def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr




def test(settings,sid=9):
    
    #sess = K.get_session()
    K.clear_session()
        #sess = tf.Session(graph=g)
        #K.set_session(sess)
    np.random.seed(sid)
    tf.random.set_seed(sid)
    
    
    
    # Model parameter
    # ----------------------------------------------------------------------------
    #           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
    # Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
    #           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
    # ----------------------------------------------------------------------------
    # ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
    # ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
    # ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
    # ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
    # ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
    # ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
    # ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
    # ---------------------------------------------------------------------------
    
    n = 3
    
    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    version = 1
    
    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2
    # Model name, depth and version
    model_type = 'ResNet%dv%d' % (depth, version)
    #sess = K.get_session()
    dset='cifar10'

    dset = settings['dset']
    batch_size = settings['batch']
    num_classes = 10
    epochs =settings['epochs']
    test_acnn = settings['test_layer']=='aconv'
    akernel_size = settings['kernel_size']
    data_augmentation = settings['data_augmentation']
    num_blocks = settings['depth']
    lr_multiplier = settings['lr_multiplier']
    nfilters=16
    normalize_data=True
    acnn_options = {'init_sigma':0.15,'norm':2,
                    'kernel_size':(akernel_size,akernel_size)}
    if dset=='mnist':
        acnn_options.update({'dropout':0.25})
    elif dset=='mnist-clut':
        normalize_data=False
        
    if 'init_sigma' in settings.keys():
        acnn_options['init_sigma']=settings['init_sigma']
    if 'norm' in settings.keys():
        acnn_options['norm']=settings['norm']
    
    if 'nfilters' in settings.keys():
        nfilters=settings['nfilters']

    

    
    ld_data = load_dataset(dset,normalize_data,options=[])
    x_train,y_train,x_test,y_test,input_shape,num_classes=ld_data
    
   
    inputs = Input(shape=input_shape)
    outputs= resnet(inputs,num_classes,num_blocks=num_blocks, 
                    kernel_size=akernel_size, num_filters=nfilters,
                    acnn=test_acnn, acnn_options=acnn_options)
    model = Model(inputs, outputs)
        
    model.summary()
    print(model_type)
        
    #lr_dict = {'all':0.001,'acnn-1/Sigma:0': 0.001,'acnn-1/Weights:0': 0.001,
    #               'acnn-2/Sigma:0': 0.001,'acnn-2/Weights:0': 0.001}
        
    lr_dict = {'all': 0.01, 'Sigma': 0.01}
    for i in lr_dict.keys():
        lr_dict[i]*=settings['lr_multiplier']
    
    MIN_SIG = 1.0/akernel_size
    MAX_SIG = akernel_size*1.0

    mom_dict = {'all':0.9}
    clip_dict = {'Sigma': [MIN_SIG, MAX_SIG]}
    decay_dict = {'all':0.1}
    e_i = x_train.shape[0] // batch_size
    
    #decay_epochs =np.array([e_i*1,e_i*2,e_i*3,e_i*4,e_i*80,e_i*120,e_i*160], dtype='int64')
    
    decay_epochs =np.array([e_i*80,e_i*120,e_i*160], dtype='int64')
    
    # WE ACTUALLY USED BELOW FOR THE TESTS OF THE PAPER 
    # WHERE WE GIVE DIFFERENT LEARNING RATE FOR SIGMA.'Sigma':0.1*lr_multiplier}
    # UNFORTUNATELY THIS DOES NOT WORK IN TF2. 
    # the paper resnet results with sgdwithcycliclr but on tf1.4
    # lr_cyclic = True 
    # if lr_cyclic:
    #     opt = SGDwithCyclicLR(peaklriter=epochs/2*e_i,lr=lr_dict, momentum = mom_dict, 
    #                           min_lr ={'all':0.0001}, #0.0001
    #                           peak_lr={'all':0.5*lr_multiplier,'Sigma':0.1*lr_multiplier},
    #                           lrsigma=0.5,
    #                           clips=clip_dict, clipvalue=1.0, verbose=2) 
    # gives 92.4 at 150 epochs for CIFAR
        

    from tensorflow_addons.optimizers.cyclical_learning_rate import CyclicalLearningRate
    
    # you can use learning schedule below. it increases the learning rate smoothly
    # until it reaches a max_lr, but for all variables. 
    def gauss_lr(epoch, lr, min_lr=1e-4, max_lr=0.5*settings['lr_multiplier'], 
                 center=epochs/3, lr_sigma=0.5):
            print("testing", epoch, lr, min_lr, max_lr, center, lr_sigma)
            lr = (min_lr + max_lr * np.exp(-(epoch-center)**2 / (center*lr_sigma)**2))
            return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(gauss_lr, verbose=1)        
        
    opt = SGD(momentum=0.9, clipvalue=1.0)
    # if dset=='lfw_faces':  does not get more than 90s
    #     print("USING ADAM")
    #     opt = AdamwithClip(1e-2, clips=clip_dict, clipvalue=1.0)
    #     red_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
    #                                                   factor=0.9, patience=10, 
    #                                                   verbose=1, mode='auto', 
    #                                                   min_delta=0.0001, 
    #                                                   cooldown=10, min_lr=1e-5)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    
    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    if test_acnn:
        model_name = dset+'_%s_acnn_resnet_model_best_sofar.h5' % model_type
    else:
        model_name = dset+'_%s_resnet_model_best_sofar.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    
    from tensorflow.keras.utils import plot_model
    #plot_model(model,'resnet_model.png', show_shapes=True, show_layer_names=False,dpi=300)
    
    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True)
    # you can try this with adam    
#    lr_reducer = ReduceLROnPlateau(factor=0.9,
#                                   cooldown=0,
#                                   patience=5,
#                                   min_lr=0.5e-6)
    
    #callbacks = [checkpoint, lr_reducer, lr_scheduler]
    stat_func_name = ['max: ', 'mean: ', 'min: ', 'var: ', 'std: ']
    stat_func_list = [np.max, np.mean, np.min, np.var, np.std]
    
    callbacks = [lr_callback] #LearningRateScheduler(pwl)
    
    MIN_SIG = 1.0/akernel_size
    MAX_SIG = akernel_size*1.0

    if  settings['test_layer']=='aconv':
        ccp1 = ClipCallback('Sigma',[MIN_SIG,MAX_SIG])
      
    silent_mode = True
    if not silent_mode:
        from keras_utils import PrintLayerVariableStats
        if test_acnn:
    
            pr_1 = PrintLayerVariableStats("lv1_blk1_res_conv1_aconv2D","Weights:0",stat_func_list,stat_func_name)
            pr_2 = PrintLayerVariableStats("lv1_blk1_res_conv1_aconv2D","Sigma:0",stat_func_list,stat_func_name)
            pr_3 = PrintLayerVariableStats("lv1_blk1_res_conv2_aconv2D","Weights:0",stat_func_list,stat_func_name)
            pr_4 = PrintLayerVariableStats("lv1_blk1_res_conv2_aconv2D","Sigma:0",stat_func_list,stat_func_name)            
            callbacks+=[pr_1,pr_2,pr_3,pr_4] #,rv_weights_1,rv_sigma_1]
        else:
            pass
    
    
    # if dset=='lfw_faces': thisis used with adam. 
    #     callbacks+=[red_lr]
        
    # Run training, with or without data augmentation.
    if not data_augmentation:
        print('Not using data augmentation.')
        his= model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # brightness_range=[0.9,1.1],
            # randomly shift images horizontally
            width_shift_range=0.2,
            # randomly shift images vertically
            height_shift_range=0.2,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            #zoom_range=[0.9,1.1],
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
    
        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
    
        # Fit the model on the batches generated by datagen.flow().
        his= model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks, 
                            steps_per_epoch=x_train.shape[0]//batch_size)
        
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score, his, model
    

def repeated_trials(settings):
    list_scores =[]
    list_histories =[]
    for i in range(settings['repeats']):
        print("--------------------------------------------------------------")
        print("REPEAT:",i)
        print("--------------------------------------------------------------")
        sc, hs, ms = test(settings,sid=31+i*17)
        list_scores.append(sc)
        list_histories.append(hs)
        
    print("Final scores", list_scores)
    mx_scores = [np.max(list_histories[i].history['val_accuracy']) for i in range(len(list_histories))]
    histories = [h.history for h in list_histories]
    print("Max sscores", mx_scores)
    print("Max stats", np.mean(mx_scores), np.std(mx_scores))
    
    import matplotlib.pyplot as plt
    val_acc = np.array([ v['val_accuracy'] for v in histories])
    mn = np.mean(val_acc,axis=0)
    st = np.std(val_acc,axis=0)
    plt.plot(mn,linewidth=2.0)
    plt.fill_between(np.linspace(0,mn.shape[0],mn.shape[0]),y1=mn-st,y2=mn+st, alpha=0.25) 
    return mx_scores, histories, list_scores






if __name__ == '__main__':
    import sys
    kwargs = {'exname':'resnet', 'dset':'mnist', 'depth':1, 
              'test_layer':'aconv',
              'epochs':1, 'batch':128, 
              'repeats':50, 'kernel_size':5, 'nfilters':16, 
              'data_augmentation':False, 'lr_multiplier':1.0,
              'init_sigma':[0.1,0.3], 'norm':2}
    # For MNIST YOU CAN USE lr_multiplier 1.0, 
    # FOR OTHER USE lr_multiplier 0.1 or lower
    
    # example run resnet-acnn-test_trials.py jun2020 lfw_faces 3 aconv 150 8 1 5 16 True 1.0
    
    #kwargs = {'exname':'resnet', 'dset':'mnist', 'depth':1, 
    #          'test_layer':'aconv',
    #          'epochs':50, 'batch':128, 
    #          'repeats':1, 'kernel_size':5, 'nfilters':16, 
    #          'data_augmentation':False, 'lr_multiplier':1.0,
    #          'init_sigma':0.1, 'norm':0}
    
    
    
    for k,v in enumerate(sys.argv):
        print(k,v)
    if len(sys.argv) > 1:
        kwargs['exname'] = sys.argv[1]
        print("exname", sys.argv[1])
    if len(sys.argv) > 2:
        kwargs['dset'] = sys.argv[2]
    if len(sys.argv) > 3:
        kwargs['depth'] = int(sys.argv[3])
    if len(sys.argv) > 4:
        kwargs['test_layer'] = sys.argv[4]
        print("test_layer ", sys.argv[4])
    if len(sys.argv) > 5:
        kwargs['epochs'] = int(sys.argv[5])
        print("epochs ", int(sys.argv[5]))
    if len(sys.argv) > 6:
        kwargs['batch'] = int(sys.argv[6])
        print("batch ", int(sys.argv[6]))
    if len(sys.argv) > 7:
        kwargs['repeats'] = int(sys.argv[7])
        print("repeating ", kwargs['repeats'], " reps")
    if len(sys.argv) > 8:
        kwargs['kernel_size'] = int(sys.argv[8])
        print("kernel_size", int(sys.argv[8]))
    if len(sys.argv) > 9:
        kwargs['nfilters'] = int(sys.argv[9])
        print("nfilters", int(sys.argv[9]))
    if len(sys.argv) > 10:
        kwargs['data_augmentation'] = sys.argv[10]=="True"
        print("data_augmentation", sys.argv[10]=="True")
    if len(sys.argv) > 11:
        kwargs['lr_multiplier'] = float(sys.argv[11])
        print("lr_multiplier", float(sys.argv[11]))
    if len(sys.argv) > 12:
        kwargs['init_sigma'] = float(sys.argv[12])
        print("init_sigma", float(sys.argv[12]))
    if len(sys.argv) > 13:
        kwargs['norm'] = int(sys.argv[13])
        print("norm", int(sys.argv[13]))

    r = repeated_trials(kwargs)
    #print(r)
    
    import time 
    delayed_start = 0 
    print("Delayed start ",delayed_start)
    time.sleep(delayed_start)
    from datetime import datetime
    now = datetime.now()
    timestr = now.strftime("%Y%m%d-%H%M%S")
    ks = str(kwargs['kernel_size'])+'x'+str(kwargs['kernel_size'])
    if not os.path.exists('outputs/resnet'):
        os.mkdir('outputs/resnet')
    filename = 'outputs/resnet/'+kwargs['exname']+'_'+kwargs['dset']+'_depth_'+str(kwargs['depth'])+'_'+kwargs['test_layer']+'_'+ks+'_'+timestr+'_'+'._results.npz'
    np.savez_compressed(filename,mod=kwargs,mx_scores =r[0], results=r)
