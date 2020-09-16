#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:19:09 2019

"""

from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Add, Lambda,Activation, \
GlobalAveragePooling2D, MaxPooling2D, Concatenate,Flatten, Dropout
from keras import regularizers, initializers, layers
from keras import backend as K
from Conv2DAdaptive_k import Conv2DAdaptive
K.clear_session()
# according to <arXiv:1512.03385> Table 6.
# Output map size       # layers        # filters
# 32x32                 2n+1            16
# 16x16                 2n              32
# 8x8                   2n              64
#
# Followed by global average pooling and a dense layer with 10 units.
# Total weighted layers: 6n+2
# Total params: 0.27M in resnet-20

momentum = 0.9
epsilon = 1e-5
weight_decay = 1e-4


def conv_layer(x,filters,kernel_size,strides,name,acnn=False,acnn_options={}):
    #print(name)
    if not acnn:
        conv = Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding='same',
            use_bias=False,
            kernel_initializer=initializers.he_normal(),
            kernel_regularizer=regularizers.l2(weight_decay),
            name=name,
            )(x)
    else:
        if not acnn_options:
            print('no options here',acnn_options)
            acnn_options = {'kernel_size':kernel_size,
                            'kernel_reg':regularizers.l2(weight_decay),
                            'init_sigma':0.15,
                            'norm':0}
        conv = Conv2DAdaptive(
            rank=2,
            nfilters=filters,
            kernel_size=acnn_options['kernel_size'],
            data_format='channels_last',
            strides=strides,
            padding='same',
            use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay),
            init_sigma=acnn_options['init_sigma'],
            norm=acnn_options['norm'],
            name=name
            )(x)

    return conv
        

def conv_bn_relu(x, filters, kernel_size, strides, name,acnn=False,acnn_options={}):
    """common conv2D block"""
    nm = '_aconv2D' if acnn  else '_conv_2D'
    print(type(name+nm), name+nm)
    x = conv_layer(x, filters, kernel_size, strides, name + nm,acnn=acnn,
                   acnn_options=acnn_options)
    x = BatchNormalization(momentum=momentum, epsilon=epsilon, name=name + '_BN')(x)
    x = Activation(activation='relu', name=name + '_relu')(x)
    return x


def conv_bn(x, filters, kernel_size, strides, name,acnn=False,acnn_options={}):
    """conv2D block without activation"""
    nm = '_aconv2D' if acnn  else 'conv_2D'
    x = conv_layer(x, filters, kernel_size, strides, name + nm,
                   acnn=acnn,acnn_options=acnn_options)
    x = BatchNormalization(momentum=momentum, epsilon=epsilon, name=name + '_BN')(x)
    return x


def res_block(x, dim,kernel_size, name,acnn=False,acnn_options={}):
    """residue block: two 3X3 conv2D stacks"""

    input_dim = int(x.shape[-1])

    # shortcut
    identity = x
    if input_dim != dim:    # option A in the original paper
        identity = MaxPooling2D(
            pool_size=(1, 1), strides=(2, 2),
            padding='same',
            name=name + '_shortcut_pool'
        )(identity)
        # BTEK modified here. 
        # origingal code was using Lambda concat with zero of same size
        #print("input_dim", input_dim, "dim", dim, "id:shape",identity.shape)
#        identity = Lambda(
#            lambda y: K.concatenate([y, K.zeros_like(y)]),
#            name=name + '_shortcut_zeropad'
#        )(identity)
#        # BTEK I concat it with itself. Only aim here is to make this same size of Residual tensor. below
        identity = Concatenate(name=name + '_shortcut_zeropad')([identity, identity])
        #print(identity.shape)

    # residual path
    res = x
    if input_dim != dim:
        res = conv_bn_relu(res, dim, (kernel_size, kernel_size), 
                           (2, 2), name + '_res_conv1',acnn=acnn,
                           acnn_options=acnn_options)
    else:
        res = conv_bn_relu(res, dim, (kernel_size, kernel_size), 
                           (1, 1), name + '_res_conv1',acnn=acnn,
                           acnn_options=acnn_options)

    res = conv_bn(res, dim, (kernel_size, kernel_size), (1, 1), 
                  name + '_res_conv2',acnn=acnn,
                  acnn_options=acnn_options)

    # add identity and residue path
    #print(identity.shape, ":+::", res.shape)
#    input("oıjoı")
    out = Add(name=name + '_add')([identity, res])
    
    out = Activation(activation='relu', name=name + '_relu')(out)
    return out


def resnet(x, num_classes, num_blocks,kernel_size,num_filters=16,acnn=False,acnn_options={}):

    # level 0:
    # input: 32x32x3; output: 32x32x16
    ##x = Dropout(0.1)(x) # remove this
    print(acnn_options)
    dropout = 0
    if 'dropout' in acnn_options.keys():
        dropout= acnn_options['dropout']
        
    x = conv_bn_relu(x, num_filters, (kernel_size, kernel_size), (1, 1), 'lv0',acnn=acnn,
                     acnn_options=acnn_options)
    
    # level 1:
    # input: 32x32x16; output: 32x32x16
    for i in range(num_blocks):
        x = res_block(x, num_filters,kernel_size, name='lv1_blk{}'.format(i + 1),
                      acnn=acnn,acnn_options=acnn_options)
    
    if dropout>0:
        x = Dropout(dropout)(x)
    # level 2:
    # input: 32x32x16; output: 16x16x32
    for i in range(num_blocks):
        x = res_block(x, num_filters*2,kernel_size, name='lv2_blk{}'.format(i + 1),
                      acnn=acnn,acnn_options=acnn_options)
    if dropout>0:
        x = Dropout(dropout)(x)
    # level 3:
    # input: 16x16x32; output: 8x8x64
    for i in range(num_blocks):
        x = res_block(x, num_filters*4,kernel_size, name='lv3_blk{}'.format(i + 1),acnn=acnn,
                      acnn_options=acnn_options)
    if dropout>0:
        x = Dropout(dropout)(x)
    # output
    x = GlobalAveragePooling2D(name='pool')(x)
    if dropout>0:
        x = Dropout(dropout)(x)
    #x = Flatten()(x)
    #x = Dense(256, name='class')(x)
    x = Dense(
        num_classes, activation='softmax',
        kernel_initializer=initializers.he_normal(),
        kernel_regularizer=regularizers.l2(weight_decay),
        name='FC'
    )(x)
    return x