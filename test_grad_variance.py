#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 16:28:46 2020

@author: btek
"""

from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Input, Dense, Dropout,\
    Flatten, BatchNormalization, AlphaDropout, Conv2D
from Conv2DAdaptive_k2 import Conv2DAdaptive
from tensorflow.keras.layers import Activation # MaxPool2D
import tensorflow.keras as keras
from tensorflow.keras import initializers, regularizers
# In[0]: MODEL
# Instantiate our linear layer (defined above) with 10 units.
#alc_layer = ALCShared(10,wshape=(784,1))
def create_simple_model(input_shape, num_classes=10, settings={}):

    act_func = 'relu'
    act_func = 'selu'
    act_func = 'relu'
    #drp_out = AlphaDropout
    drp_out = Dropout
    #from keras.regularizers import l2
    
    node_in = Input(shape=input_shape, name='inputlayer')

    #node_fl = Flatten(data_format='channels_last')(node_in)
    
    node_ = Dropout(0.01)(node_in)
    heu= keras.initializers.he_uniform
    h = 1 
    afw = settings['kernel_size']
    for nh in settings['nhidden']:
        if settings['neuron']=='alc':
            node_ = Conv2DAdaptive(rank=2,nfilters=nh,
                                kernel_size=(afw,afw),
                                data_format='channels_last',strides=1,
                                padding='same',name='aconv-'+str(h), activation='linear',
                                trainSigmas=True, trainWeights=True,
                                init_sigma=[0.1,0.3],
                                gain = 1.0,
                                kernel_regularizer=None,
                                sigma_regularizer=None,#regularizers.l2(1e-5),
                                init_bias=initializers.Constant(0),
                                norm=2,use_bias=False)(node_)
        else:
            node_ = Conv2D(nh, kernel_size=(afw,afw),
                           name='conv-'+str(h),activation='linear',
                          kernel_initializer=heu())(node_)
        
        node_ = BatchNormalization()(node_)
        node_ = Activation(act_func)(node_)
        node_ = drp_out(0.25)(node_)
        h = h + 1
        
    node_ = Flatten(data_format='channels_last')(node_)
    node_ = Dense(num_classes, name='class', activation='relu', 
                     kernel_initializer=keras.initializers.he_uniform(),
                    kernel_regularizer=None)(node_)
    node_fin = Dense(num_classes, name='softmax', activation='softmax', 
                     kernel_initializer=keras.initializers.he_uniform(),
                    kernel_regularizer=None)(node_)
    
    model = Model(inputs=node_in, outputs=[node_fin])
    
    return model

# LOAD DATA
# In[0]: DATA
import tensorflow as tf
# Prepare a dataset.
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

BATCHSIZE=256

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(BATCHSIZE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCHSIZE)

# In[1]: CREATE MODEL
# Create an instance of the model
settings={}
settings['nhidden']=[32,32]
settings['kernel_size'] = 7
settings['neuron']='conv'
model = create_simple_model(x_train.shape[1:], 10, settings)
model.summary()
#alc_layer = keras.layers.Dense(10)

# Instantiate a logistic loss function that expects integer targets.
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.SGD(1e-1, momentum=0.9,clipvalue=1.0)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')



# In[1]: TRAINING AND TEST FUNCTIONS
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)
  
  
@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)



# In[1]: TRAIN 

EPOCHS = 15

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch + 1,
                        train_loss.result(),
                        train_accuracy.result() * 100,
                        test_loss.result(),
                        test_accuracy.result() * 100))

