#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 21:22:43 2019

@author: btek
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"
import plot_utils as pu
#pu.paper_fig_settings(addtosize=2)
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
tf2 = False
if tf2:
    from tensorflow.keras.datasets import mnist,fashion_mnist, cifar10
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D
    from tensorflow.keras import backend as K
    from tensorflow.keras.losses import mse
    from keras_utils_tf2 import WeightHistory as WeightHistory
    from keras_utils_tf2 import RecordWeights, PrintLayerVariableStats, SGDwithLR
else:
    from keras.datasets import mnist,fashion_mnist, cifar10
    from keras.models import Sequential, Model
    from keras.layers import Input, Dense, Dropout, Flatten, Conv2D
    from keras import backend as K
    from keras.losses import mse
    from keras_utils import WeightHistory as WeightHistory
    from keras_utils import RecordOutput, RecordWeights,\
    PrintLayerVariableStats, SGDwithLR, AdamwithClip
    from keras.regularizers import l2, l1, l1_l2
from Conv2DAdaptive_k import Conv2DAdaptive
import matplotlib.pyplot as plt
from skimage import io, filters
from scipy import ndimage,misc


K.clear_session()
sid = 9
np.random.seed(sid)
tf.random.set_random_seed(sid)
tf.compat.v1.random.set_random_seed(sid)

from datetime import datetime
now = datetime.now()
timestr = now.strftime("%Y%m%d-%H%M%S")
print(timestr)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
inputimg = x_train[0]/255
sh = (inputimg.shape[0],inputimg.shape[1],1)
outputimages = np.zeros(shape=[inputimg.shape[0],inputimg.shape[1],9],dtype='float32')
k3 = np.array([[-1,2,-1],]).repeat(3,axis=0).T
k5 = np.array([[-1,-1,4,-1,-1],]).repeat(5,axis=0).T
k9 = np.array([[-1,-1,-1,-1,8,-1,-1,-1,-1],]).repeat(9,axis=0).T
outputimages[:,:,0] = filters.laplace(inputimg, ksize=3)
outputimages[:,:,1] = filters.sobel_h(inputimg)
outputimages[:,:,2] = filters.sobel_v(inputimg)
outputimages[:,:,3] = filters.gaussian(inputimg,sigma=0.25)
outputimages[:,:,4] = filters.gaussian(inputimg,sigma=1.0)
outputimages[:,:,5] = filters.gaussian(inputimg,sigma=2.0)
outputimages[:,:,6] = filters.laplace(outputimages[:,:,5], ksize=3)
outputimages[:,:,7] = filters.sobel_h(outputimages[:,:,5])
outputimages[:,:,8] = filters.sobel_v(outputimages[:,:,5])
#outputimages[:,:,0] = ndimage.correlate(inputimg, k3,mode='constant',cval=0)
#outputimages[:,:,1] = ndimage.correlate(inputimg, k5,mode='constant',cval=0)
#outputimages[:,:,2] = ndimage.correlate(inputimg, k9,mode='constant',cval=0)
#outputimages[:,:,3] = filters.gaussian(inputimg,sigma=0.25)
#outputimages[:,:,4] = filters.gaussian(inputimg,sigma=0.5)
#outputimages[:,:,5] = filters.gaussian(inputimg,sigma=1.0)
#outputimages[:,:,6] = ndimage.correlate(outputimages[:,:,3],k3)
#outputimages[:,:,7] = ndimage.correlate(outputimages[:,:,4],k5)
#outputimages[:,:,8] = ndimage.correlate(outputimages[:,:,5],k9)

y = y_train[0]

node_in = Input(shape=sh, name='inputlayer')
# smaller initsigma does not work well.
nf = outputimages.shape[-1]
acnn = True
EPOCHS=2500
reg = 1e-3
fs = 9
if acnn:
    node_acnn = Conv2DAdaptive(rank=2,nfilters=nf,kernel_size=(fs,fs),
                             data_format='channels_last',
                             strides=1,
                             padding='same',name='acnn',activation='linear',
                             init_sigma=0.25, trainSigmas=True,
                             trainWeights=True,
                             kernel_regularizer=l2(reg),
                             norm=2,
                             sigma_regularizer=None)(node_in)
else:

    node_acnn = Conv2D(filters=nf,kernel_size=(fs,fs),
                             data_format='channels_last',
                             padding='same',name='acnn',
                             activation='linear',
                             kernel_regularizer=l2(reg))(node_in)


model = Model(inputs=node_in, outputs=[node_acnn])

   # model.summary()

from lr_multiplier import LearningRateMultiplier
from keras.optimizers import SGD, Adadelta
lr_dict = {'all':0.1,'acnn/Sigma:0': 0.1,'acnn/Weights:0': 0.1}

mom_dict = {'all':0.9,'acnn/Sigma:0': 0.9,'acnn/Weights:0': 0.9}
clip_dict = {'acnn/Sigma:0': [0.10, 2.0]}
# WORKS WITH OWN INIT With glorot uniform
#print("WHAT THE ", lr_dict)
opt = SGDwithLR(lr=lr_dict, momentum = mom_dict, 
                clips=clip_dict,clipvalue=1.0)

#opt = AdamwithClip(clips=clip_dict,clipnorm=1.0)
#(lr_dict, mom_dict)


def customloss(lay):
    def loss(y_true, y_pred):
        # Use x here as you wish
        err = mse(y_true,y_pred)
        #a = K.mean(K.square(y_pred - y_true), axis=-1)
        b= K.mean(K.square(lay.W*(1.0 - lay.U())))
        c= K.mean(lay.Sigma)
        mn_a = K.max(err)
        mn = tf.zeros_like(mn_a)
        print("Shape of a",err.shape)
        #err+= tf.clip_by_value(b, mn, mn_a*2.0)
        #err+= 1e-2*c
        err+= tf.clip_by_value(K.mean(K.square(lay.W)), mn, mn_a*1.0)
        err+= tf.clip_by_value(c, mn, mn_a*0.7)
        #err += 1e-2*K.mean(K.abs(lay.W))
        return err

    return loss


def mseloss(y_true,y_pred):
    return mse(y_true,y_pred)


model.compile(loss='mse',#customloss(model.get_layer('acnn'))
              optimizer=opt, metrics=[mseloss])
model.summary()


inputimg2 = np.expand_dims(np.expand_dims(inputimg,axis=0), axis=3)
outputimages2 = np.expand_dims(outputimages,axis=0)

from keras.utils import plot_model
plot_model(model,'figures/may2020/auto_enc.png', show_shapes=True, show_layer_names=False)
#print info about weights

acnn_layer = model.get_layer('acnn')
all_params=acnn_layer.weights
print("All params:",all_params)
acnn_params = acnn_layer.get_weights()
for i,v in enumerate(all_params):
    print(v, ", max, mean", np.max(acnn_params[i]),np.mean(acnn_params[i]),"\n")


mchkpt = keras.callbacks.ModelCheckpoint('models/weights.txt', monitor='val_loss', 
                                         verbose=0, save_best_only=False, 
                                         save_weights_only=False, 
                                         mode='auto', period=1)
wh0= WeightHistory(model, "acnn")

sigma_history = []
sigma_call = lambda x, batch=1, logs={}: x.append(acnn_layer.get_weights()[0])

cov_dump = keras.callbacks.LambdaCallback(on_epoch_end=sigma_call(sigma_history))

cb=[]
if acnn:
    rv = RecordWeights("acnn","acnn/Sigma:0",max_record=2000, record_batches=True)
    cb = [wh0,cov_dump,rv]

#rv = np.ones(shape=(1000,))
params_before = acnn_layer.get_weights()

xticks = np.linspace(1,fs,3, dtype='int')

def plotoneline(vec):
    pass
#pu.paper_fig_settings(-2)
if acnn:
    num_images=min(nf,12)
    fig=plt.figure(figsize=(num_images*3,10))
    
    U_before = K.eval(acnn_layer.calc_U())
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(np.float32(U_before[:,:,0,i]), cmap=plt.cm.bone)
        #plt.title("U before training func")
        if i==0:
            plt.yticks([0,4,8])
            plt.xticks([0,4,8])
        else:
            plt.axis('off')
        #plt.xticks(ticks=xticks)
        #plt.yticks(ticks=xticks)
    
    plt.show()

history = model.fit(inputimg2, outputimages2,
          batch_size=1,
          epochs=EPOCHS,
          verbose=0, callbacks=cb)

plt.figure()
pu.paper_fig_settings(8)
plt.plot(history.history['mseloss'])
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.xlim([-10,200])
plt.grid()

#plt.title("Loss")
print( "Final loss", history.history['mseloss'][-1] )

if acnn:
    np.save('test_filters_acnn_loss',history.history['mseloss'])
    pu.paper_fig_settings(4)
    print ("RECORD", len(rv.record), rv.record[0].shape)
    rv_arr = np.array(rv.record)
    plt.figure()
    plt.plot(rv_arr)
    plt.ylabel('Variance $\sigma_s$')
    plt.xlabel('Epoch')
    plt.xlim([-10,500])
    plt.grid()
    #plt.title("Sigma")
else:
    np.save('test_filters_cnn_loss',history.history['mseloss'])

try:
    last_acnn = np.load('test_filters_acnn_loss.npy')
    last_cnn = np.load('test_filters_cnn_loss.npy')
    plt.figure()
    pu.paper_fig_settings(4)
    plt.plot(last_acnn,'r-')
    plt.plot(last_cnn,'b--')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.xlim([-10,500])
    plt.grid()
    plt.legend(['AConv','Conv'])
except:
    print('no files saved yet!')


#print info about weights
print("After training:",all_params)
params_after = acnn_layer.get_weights()
for i,v in enumerate(all_params):
    print(v, ", max, mean", np.max(params_after[i]),np.mean(params_after[i]),"\n")

# print info about sigmas
#print("Recorded sigma history", sigma_history)
#print("Recorded weight history", wh0.get_epochlist())

# In[pltos]
pred_images = model.predict(inputimg2,  verbose=1)
print("Prediction shape",pred_images.shape)
pltthis = True
if pltthis:
    if not acnn:
        plt.figure(figsize=(12,6))
    
        for i in range(0,nf):
            plt.subplot(1, nf, i+1)
            plt.imshow(np.float32(acnn_params[0][:,:,0,i]))
    #print("U -shape: ", acnn_layer.U().shape,type(K.eval(acnn_layer.U()[:,:,0,0])))
    #print("Prod-shape", (acnn_params[1][:,:,0,0]*acnn_layer.U()[:,:,0,0]).shape)    
    #plt.subplot(1, nf, 2)
    
        plt.show()    
    else:
        print("Plotting kernels before...")
        num_images=min(pred_images.shape[3],12)
        
        # plot input image 
        fig=plt.figure(figsize=(num_images*3,10))
        plt.subplot(1,num_images, 1)
        plt.imshow(np.squeeze(inputimg2[0,:,:,0]), plt.cm.gray)
        plt.axis('off')
        
        # plot target images 
        fig=plt.figure(figsize=(num_images*3,10))
        labels =['Laplace$_{3x3}$', 'Sobel$_h$', 'Sobel$_v$',
                 'Gauss$_{\sigma=.25}$','Gauss$_{\sigma=.5}$',
                 'Gauss$_{\sigma=1.0}$',
                 'Gs$_{\sigma=0.5}$+Lpl$_{3x3}$',
                 'Gs$_{\sigma=.5}$+Sbl$_h$',
                 'Gs$_{\sigma=.5}$+Sbl$_v$']
        for i in range(num_images):
            plt.subplot(1,num_images, i+1)
            plt.imshow(np.squeeze(outputimages2[0,:,:,i]),cmap=plt.cm.bone)
            #plt.title("target image")
            print("Max-in:",i," ",np.max(np.squeeze(outputimages2[0,:,:,i])))
            if i==0:
                plt.yticks([0,13,27])
                plt.xticks([0,13,27])
            else:
                plt.axis('off')
            plt.title(labels[i])
        plt.show()
        
        # plot initial filters. 
        
        # plot predicted images
        fig=plt.figure(figsize=(num_images*3,10))
        plt.subplot(1,num_images, 1)
        plt.imshow(np.squeeze(inputimg2[0,:,:,0]))
        for i in range(num_images):
            plt.subplot(1,num_images, i+1)
            plt.imshow(np.squeeze(pred_images[0,:,:,i]), cmap=plt.cm.bone)
            #plt.title("pred image")
            print("MAx:","pred",i,np.max(np.squeeze(pred_images[0,:,:,i])))
            if i==0:
                plt.yticks([0,13,27])
                plt.xticks([0,13,27])
            else:
                plt.axis('off')
            
        plt.show()
        
        # plot filters after training
        plt.figure(figsize=((num_images*3,10)))
        for i in range(num_images):
            plt.subplot(1, num_images+1, i+1)
            #print(acnn_params[1].shape)
            plt.imshow(np.squeeze(params_before[1][:,:,0,i]), plt.cm.bone)
            #print("MAx:","pred",i,np.max(np.squeeze(acnn_params[i])))
            if i==0:
                plt.yticks([0,4,8])
                plt.xticks([0,4,8])
            else:
                plt.axis('off')
                
        #plt.colorbar()
        plt.show()
        
         # plot filters after training
        plt.figure(figsize=((num_images*3,10)))
        for i in range(num_images):
            plt.subplot(1, num_images+1, i+1)
            #print(acnn_params[1].shape)
            plt.imshow(np.squeeze(params_after[1][:,:,0,i]), plt.cm.bone)
            #print("MAx:","pred",i,np.max(np.squeeze(acnn_params[i])))
            if i==0:
                plt.yticks([0,4,8])
                plt.xticks([0,4,8])
            else:
                plt.axis('off')
        #plt.colorbar()
        plt.show()
    
    
        plt.figure(figsize=((num_images*3,10)))
        #plt.title("U after training func")
        for i in range(num_images):
            plt.subplot(1, num_images+1, i+1)
            plt.imshow(np.float32(K.eval(acnn_layer.calc_U()[:,:,0,i])),plt.cm.bone)
            if i==0:
                plt.yticks([0,4,8])
                plt.xticks([0,4,8])
            else:
                plt.axis('off')
        plt.show()
            #plt.colorbar()
        plt.figure(figsize=((num_images*3,10)))
        #plt.title("UW after training func")
        for i in range(num_images):
            plt.subplot(1, num_images+1, i+1)
            plt.imshow(np.float32(params_after[1][:,:,0,i]*K.eval(acnn_layer.calc_U()[:,:,0,i])), plt.cm.bone)
            if i==0:
                plt.yticks([0,4,8])
                plt.xticks([0,4,8])
            else:
                plt.axis('off')
        plt.show() 
            
    
    
    
        #print( model.get_layer('acnn').output )
        print( "Final Sigmas", model.get_layer('acnn').get_weights()[0] )
        print( "Final loss", history.history['loss'][-1] )
    
        #K.clear_session()
        pu.paper_fig_settings(4)
        plt.figure(figsize=(12,10))
        #print("U -shape: ", acnn_layer.U().shape,type(K.eval(acnn_layer.U()[:,:,0,0])))
        #print("Prod-shape", (acnn_params[1][:,:,0,0]*acnn_layer.U()[:,:,0,0]).shape)
    
        plt.subplot(1, 3, 1)
        plt.imshow(np.float32(params_after[1][:,:,0,i]),plt.cm.bone)
        plt.yticks([0,4,8])
        plt.xticks([0,4,8])
        plt.subplot(1, 3, 2)    
        plt.imshow(np.float32(K.eval(acnn_layer.calc_U()[:,:,0,i])),plt.cm.bone)
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(np.float32(params_after[1][:,:,0,i]*K.eval(acnn_layer.calc_U()[:,:,0,i])),plt.cm.bone)
        plt.axis('off')
        
        #plt.title("U*Weights")
        
                #plt.imshow()
        
                #print("MAx:","pred",i,np.max(np.squeeze(acnn_params[i])))
            #fig.colorbar(im, ax=ax1)
        plt.show()
        
        
        pu.paper_fig_settings(4)
        plt.figure(figsize=(12,10))
            #print("U -shape: ", acnn_layer.U().shape,type(K.eval(acnn_layer.U()[:,:,0,0])))
            #print("Prod-shape", (acnn_params[1][:,:,0,0]*acnn_layer.U()[:,:,0,0]).shape)    
        plt.subplot(1, 3, 1)
        plt.imshow(np.float32(params_after[1][:,:,0,2]),plt.cm.bone)
        plt.yticks([0,4,8])
        plt.xticks([0,4,8])
        plt.subplot(1, 3, 2)    
        plt.imshow(np.float32(K.eval(acnn_layer.calc_U()[:,:,0,2])),plt.cm.bone)
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(np.float32(params_after[1][:,:,0,2]*K.eval(acnn_layer.calc_U()[:,:,0,2])),plt.cm.bone)
        plt.axis('off')
    #plt.title("U*Weights")
    
            #plt.imshow()
    
            #print("MAx:","pred",i,np.max(np.squeeze(acnn_params[i])))
        #fig.colorbar(im, ax=ax1)
    plt.show()
