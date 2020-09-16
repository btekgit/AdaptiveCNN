#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:22:35 2020

@author: original code from fchollet.  modified by BTEK
"""

"""
Title: Image segmentation with a U-Net-like architecture
Author: [fchollet](https://twitter.com/fchollet)

Date created: 2019/03/20
Last modified: 2020/04/20
Description: Image segmentation model trained from scratch on the Oxford Pets dataset.
"""
"""
## Download the data
"""

"""shell
curl -O http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
curl -O http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xf images.tar.gz
tar -xf annotations.tar.gz
"""

"""
## Prepare paths of input images and target segmentation masks
"""

import os

if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
    os.environ['CUDA_VISIBLE_DEVICES']="0"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"


# download the data first or choose the appropriate directory
root = '/home/btek/datasets/oxfordpets/'
root = '/media/home/rdata/oxfordpets/'
root = '/home/btek/datasets/oxfordpets/'
input_dir = root+"images/"
target_dir = root+"annotations/trimaps/"

import tensorflow as tf
import numpy as np
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from IPython.display import clear_output
import matplotlib.pyplot as plt

"""## Download the Oxford-IIIT Pets dataset

The dataset is already included in TensorFlow datasets, all that is needed to do is download it. The segmentation masks are included in version 3+.
"""

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
IMG_SIZE = (128,128)
OUTPUT_CHANNELS = 3
"""The following code performs a simple augmentation of flipping an image. In addition,  image is normalized to [0,1]. Finally, as mentioned above the pixels in the segmentation mask are labeled either {1, 2, 3}. For the sake of convenience, let's subtract 1 from the segmentation mask, resulting in labels that are : {0, 1, 2}."""

# In[]:
    
def normalize(input_image, input_mask):
  input_image = (tf.cast(input_image, tf.float32) / 255.0) * 4.0 - 2.0
  input_mask -= 1
  return input_image, input_mask

@tf.function
def load_image_train(datapoint):
  p = tf.random.uniform(())
  
  input_image = tf.image.resize(datapoint['image'], IMG_SIZE,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  input_mask = tf.image.resize(datapoint['segmentation_mask'], IMG_SIZE,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    

  if p > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)
  if p > 0.5:
     input_image = tf.image.random_brightness(input_image, 0.2)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'],IMG_SIZE)
  input_mask = tf.image.resize(datapoint['segmentation_mask'], IMG_SIZE)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

"""The dataset already contains the required splits of test and train and so let's continue to use the same split."""

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 32
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

"""Let's take a look at an image example and it's correponding mask from the dataset."""
from PIL import Image


def display(display_list, fname=None):
  plt.figure(figsize=(15, 15))

  title = ['input', 'gt', 'pred']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    #plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
    if fname:
        im = tf.keras.preprocessing.image.array_to_img(display_list[i])
        im.save(fname+'_'+title[i]+'.png')
   
  plt.show()
  

for image, mask in train.take(1):
  sample_image, sample_mask = image, mask
display([sample_image, sample_mask])


def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None, num=1, fname=None):
  if dataset:
    k=1
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)],fname+'_'+str(k))
      k+=1
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

# In[]:
    
"""## Define the model
The model being used here is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). In-order to learn robust features, and reduce the number of trainable parameters, a pretrained model can be used as the encoder. Thus, the encoder for this task will be a pretrained MobileNetV2 model, whose intermediate outputs will be used, and the decoder will be the upsample block already implemented in TensorFlow Examples in the [Pix2pix tutorial](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py). 

The reason to output three channels is because there are three possible labels for each pixel. Think of this as multi-classification where each pixel is being classified into three classes.
"""
#   return tf.keras.Model(inputs=inputs, outputs=x)
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import initializers, regularizers
from Conv2DAdaptive_k2 import Conv2DAdaptive, Conv2DTransposeAdaptive
from keras_utils_tf2 import SGDwithLR, ClipCallback, RenormCallback
import matplotlib.pyplot as plt




def get_model(img_size, num_classes,  options=[]):
    
    inputs = keras.Input(shape=img_size + (3,))

    ### dropout does not work on the  input
    # if options['drop_out']>0.0:
    #     x = keras.layers.Dropout(options['drop_out'])(inputs)
    # else:
    x = inputs
    # Entry block
    #f1 = [64,128,256]
    f1 = [48,64,96]
    #f1 = [8,16,32]
    f2 = [96, 64, 48, 32]
    #f2 = [32, 16, 8, 32]
    initer = 'glorot_uniform'
    if options['acnn']:
        print(options)
        x = Conv2DAdaptive(filters=32,
                            kernel_size=options['kernel_size'],
                            data_format='channels_last',
                            strides=options['strides'],
                            padding='same',
                            use_bias=False,
                            trainSigmas=True,
                            kernel_regularizer=options['kernel_reg'],
                            sigma_regularizer= options['sigma_reg'],
                            init_sigma=options['init_sigma'],
                            norm=options['norm'],
                            name='aconv2d',
                            activation="linear",
                            kernel_initializer=initer,
                            )(inputs)
    else:
        x = layers.Conv2D(32, options['kernel_size'], strides=2, 
                          use_bias=False, kernel_initializer=initer, padding="same")(x)


    x = layers.BatchNormalization()(x)
    x = layers.Activation(options['act'])(x)

    previous_block_activation = x  # Set aside residual        
    
    
    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in f1:
        x = layers.Activation(options['act'])(x)
        x = layers.SeparableConv2D(filters, 3, kernel_initializer=initer, padding="same")(x)
        x = layers.BatchNormalization()(x)
        

        x = layers.Activation(options['act'])(x)
        x = layers.SeparableConv2D(filters, 3, kernel_initializer=initer, padding="same")(x)
        x = layers.BatchNormalization()(x)
    
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        
        
        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, 
                                 kernel_initializer=initer, padding="same")(
                                     previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual


    if options['drop_out']>0.0:
        x = keras.layers.Dropout(options['drop_out'])(x)
    

    ### [Second half of the network: upsampling inputs] ###

    for filters in f2:
        x = layers.Activation(options['act'])(x)
        x = layers.Conv2DTranspose(filters, 3,kernel_initializer=initer, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation(options['act'])(x)
        x = layers.Conv2DTranspose(filters, 3,kernel_initializer=initer, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, kernel_initializer=initer, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    #outputs = layers.Conv2D(num_classes, 3, activation="linear", padding="same")(x)
    
    outputs = layers.Conv2D(num_classes, 3, activation="linear", 
                            kernel_initializer=initer, 
                            kernel_regularizer=tf.keras.regularizers.l2(0),
                            padding="same")(x)
    

    # Define the model
    model = keras.Model(inputs, outputs)
    return model



"""## Train the model
Now, all that is left to do is to compile and train the model. The loss being used here is `losses.SparseCategoricalCrossentropy(from_logits=True)`. The reason to use this loss function is because the network is trying to assign each pixel a label, just like multi-class prediction. In the true segmentation mask, each pixel has either a {0,1,2}. The network here is outputting three channels. Essentially, each channel is trying to learn to predict a class, and `losses.SparseCategoricalCrossentropy(from_logits=True)` is the recommended loss for 
such a scenario. Using the output of the network, the label assigned to the pixel is the channel with the highest value. This is what the create_mask function is doing.
"""   
settings = {'repeats':5, 'acnn':True, 'Epochs':75, 'batch_size':32, 'lr_all':0.01, 'lr_sigma':0.0001,
             'kernel_size':7, 'init_sigma':[0.1,0.25], 'strides':2,'norm':2,
             'drop_out':0,
             'act':'elu',   'kernel_reg':regularizers.l2(0),
                             'sigma_reg': regularizers.l2(1e-10),
                             'debug':False}

# settings for acnn are init_sigma, norm, lr_sigma

mod = ['conv','aconv']
mod = mod[int(settings['acnn'])]
history_list =[]
for r in range(settings['repeats']):
    model = get_model(IMG_SIZE,OUTPUT_CHANNELS, settings)
    lr_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=2)

    #opt = tf.keras.optimizers.SGD(learning_rate=settings['lr_all'], momentum=0.5, clipvalue=1.0) # SGD reaches 88~
    opt = tf.keras.optimizers.Adam(learning_rate=settings['lr_all'], clipvalue=1.00) # 0.001 works ok with clip
    #opt = tf.keras.optimizers.SGD(learning_rate=settings['lr_all'], momentum=0.9, clipvalue=1.0) # 0.001 works ok with clip

    #opt = tfa.optimizers.RectifiedAdam(lr=settings['lr_all'], clipvalue=1.00)
    #opt = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

    model.compile(optimizer = opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    """Have a quick look at the resulting model architecture:"""
    
    model.summary()
    tf.keras.utils.plot_model(model, 'unet_model.png', show_shapes=True)
    
    
    """Let's try out the model to see what it predicts before training."""
    
    
    """Let's observe how the model improves while it is training. To accomplish this task, a callback function is defined below."""
    MIN_SIG = 1.0/settings['kernel_size']
    MAX_SIG = 5.0
    ccp1 = ClipCallback('Sigma',[MIN_SIG,MAX_SIG])
    #rncb = RenormCallback('aconv2d/Weights')

    mcpnt = tf.keras.callbacks.ModelCheckpoint('best_weights'+mod+'.h5',
                                       monitor='val_accuracy', 
                                       save_best_only=True, 
                                       save_weights_only=True)
    
    
    BATCH_SIZE = settings['batch_size']
    EPOCHS = settings['Epochs']
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS
    
    batches = [BATCH_SIZE, BATCH_SIZE//2,BATCH_SIZE//4,BATCH_SIZE//8,BATCH_SIZE//16]
    epoches = [EPOCHS//4,EPOCHS//4,EPOCHS//4,EPOCHS//4,EPOCHS//4]
    
    model_history = model.fit(train_dataset, epochs=EPOCHS,
                                  steps_per_epoch=STEPS_PER_EPOCH,
                                  validation_steps=VALIDATION_STEPS,
                                  
                                  validation_data=test_dataset,
                                  callbacks=[ccp1,mcpnt,lr_rate])    
    
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    
    history_list.append(model_history.history)

model.load_weights('best_weights'+mod+'.h5')


plt.figure()
plt.plot( loss, 'r', label='Training loss')
plt.plot( val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

import time
import numpy as np
time.sleep(0)
from datetime import datetime
now = datetime.now()
timestr = now.strftime("%Y%m%d-%H%M%S")
ks = str(settings['kernel_size'])+'x'+str(settings['kernel_size'])+'_Ep_'+str(settings['Epochs'])+'_Nodrp_'
if not os.path.exists('outputs/cat_dog_data'):
    os.mkdir('outputs/cat_dog_data')
if settings['acnn']:
    filename = 'outputs/cat_dog_data/cat_dog_segmentation_aconv_'+ks+timestr+'_'+'._results.npz'
else:
    filename = 'outputs/cat_dog_data/cat_dog_segmentation_conv_'+ks+timestr+'_'+'._results.npz'
    
np.savez_compressed(filename, mod=settings, history=history_list)


# In[]: Display output



show_predictions(test_dataset, 3,filename[:-14]+'_output')

# In[]: results
import plot_utils as pu
pu.paper_fig_settings(addtosize=8)
import os
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt

conv_7_val_acc = np.load('outputs/cat_dog_data/cat_dog_segmentation_conv_7x7_Ep_75_Nodrp_20200831-203618_._results.npz', allow_pickle=True)['history']
conv_3_val_acc = np.load('outputs/cat_dog_data/cat_dog_segmentation_conv_3x3_Ep_75_Nodrp_20200830-205225_._results.npz', allow_pickle=True)['history']
aconv_val_acc = np.load('outputs/cat_dog_data/cat_dog_segmentation_aconv_7x7_Ep_75_Nodrp_20200831-232842_._results.npz', allow_pickle=True)['history']

'''ACONV RESULTS 86.84 max, 86.05 mean 
settings repeats': 5, 'acnn': True, 'Epochs': 75, 'batch_size': 32, 
'lr_all': 0.01, 'lr_sigma': 0.0001, 'kernel_size': 7, 
'init_sigma': [0.1, 0.5], 'strides': 2, 'norm': 2, 
'drop_out': 0, 'act': 'elu', 
'kernel_reg': 0, 
'sigma_reg': 1e-10, 'debug': False}

Paper results                                         
same as above but init_sigma:[0.1, 0.25]
Aconv 7 Max: 0.8661 Mean: 0.8621 Std: 0.003
Conv 7 mx: Max: 0.8574 Mean: 0.8554 Std: 0.003
Conv 3 mx: Max: 0.8632 Mean: 0.8589 Std: 0.003
Conv 7: Ttest_indResult(statistic=3.6757874608453442, pvalue=0.006257035742880989)
Conv 3: Ttest_indResult(statistic=1.695854588934318, pvalue=0.1283554989434309)
'''

his_tit = 'val_accuracy'
aconv_list=np.zeros((5,len(aconv_val_acc[0][his_tit])))
conv_7_list=np.zeros((5,len(conv_7_val_acc[0][his_tit])))
conv_3_list=np.zeros((5,len(conv_3_val_acc[0][his_tit])))

for i in range(5):
    aconv_list[i,:] = aconv_val_acc[i][his_tit]
    conv_7_list[i,:] = conv_7_val_acc[i][his_tit]
    conv_3_list[i,:] = conv_3_val_acc[i][his_tit]

template = 'Max: {:.4f} Mean: {:.4f} Std: {:.3f}'
print("Aconv 7 "+template.format(np.max(np.max(aconv_list,axis=1),axis=0),

      np.mean(np.max(aconv_list,axis=1),axis=0), 
      np.std(np.max(aconv_list,axis=1),axis=0))) 
print("Conv 7 mx: "+template.format(np.max(np.max(conv_7_list,axis=1),axis=0), 
      np.mean(np.max(conv_7_list,axis=1),axis=0),
      np.std(np.max(conv_7_list,axis=1),axis=0)))

print("Conv 3 mx: "+template.format(np.max(np.max(conv_3_list,axis=1),axis=0), 
      np.mean(np.max(conv_3_list,axis=1),axis=0), 
      np.std(np.max(conv_3_list,axis=1),axis=0)))


print("Conv 7:",ttest_ind(np.max(aconv_list,axis=1), np.max(conv_7_list,axis=1)))
print("Conv 3:", ttest_ind(np.max(aconv_list,axis=1), np.max(conv_3_list,axis=1)))


mn = np.mean(aconv_list, axis=0)
mx = np.max(aconv_list, axis=0)
mnn = np.min(aconv_list, axis=0)

plt.figure(figsize=(8,6))
plt.plot(np.mean(aconv_list, axis=0), linewidth=3)
plt.plot(np.mean(conv_7_list, axis=0), '--',linewidth=3)
plt.plot(np.mean(conv_3_list, axis=0), linewidth=3)
plt.fill_between(np.linspace(0,mn.shape[0],mn.shape[0]), y1=mx, y2=mnn, alpha=0.35) 

mn = np.mean(conv_7_list, axis=0)
mx = np.max(conv_7_list, axis=0)
mnn = np.min(conv_7_list, axis=0)
plt.fill_between(np.linspace(0,mn.shape[0],mn.shape[0]), y1=mx, y2=mnn, alpha=0.1) 
#plt.ylim([0.82, 0.87])
plt.ylim([0.38, 0.75])
plt.grid()
plt.legend(['aconv_7x7', 'conv_7x7','conv_3x3'])
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.show()



