#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 19:07:34 2018
LAst update Jun 18 2019
# mnist working
after 2 epochs
Test loss: 0.05096351163834333
Test accuracy: 0.9837

@author: btek
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']="1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"

from keras import backend as K
from keras.engine.topology import Layer
from keras.utils import conv_utils
from keras import activations, regularizers, constraints
from keras import initializers
from keras.engine import InputSpec
import numpy as np
import tensorflow as tf
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
from keras.datasets import mnist,cifar10, fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Add, ReLU
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization,Activation, Concatenate
from keras.regularizers import l2,l1
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras_preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adadelta
#from keras.initializers import glorot_uniform as w_ini
from keras.initializers import he_uniform as w_ini
from keras.initializers import VarianceScaling as VS_ini
from keras import backend as K
from keras_utils import RecordVariable, \
PrintLayerVariableStats, SGDwithLR, AdamwithClip, PrintAnyOutputVariable


#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

def sqrt32(x):
    return np.sqrt(x,dtype='float32')

def idx_init(shape, dtype='float32'):
    idxs = np.zeros((shape[0], shape[1]),dtype)
    c = 0
    # assumes square filters

    wid = np.int(np.sqrt(shape[0]))
    hei =np.int(np.sqrt(shape[0]))
    f = np.float32
    for x in np.arange(wid):  # / (self.incoming_width * 1.0):
        for y in np.arange(hei):  # / (self.incoming_height * 1.0):
            idxs[c, :] = np.array([x/f(wid-1), y/f(hei-1)],dtype)
            c += 1

    return idxs

def cov_init(shape, dtype='float32'):

    cov = np.identity(shape[1], dtype)
    # shape [0] must have self.incoming_channels * self.num_filters
    cov = np.repeat(cov[np.newaxis], shape[0], axis=0)

    #for t in range(shape[0]):
    #    cov[t] = cov[t]
    return cov



class Conv2DAdaptiveDyn(Layer):
    def __init__(self, rank, nfilters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=False,
                 kernel_regularizer=None,
                 sigma_regularizer=None,
                 gain=1.0,
                 init_sigma=0.1,
                 init_w=initializers.glorot_uniform(),
                 init_bias = initializers.Constant(0.0),
                 trainSigmas=True,
                 trainWeights=True,
                 trainGain=False,
                 reg_bias=None,
                 norm = 0,
                 **kwargs):
        super(Conv2DAdaptiveDyn, self).__init__(**kwargs)
        #def __init__(self, num_filters, kernel_sigmaze, incoming_channels=1, **kwargs):
        self.rank = rank
        self.nfilters = nfilters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = data_format
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.gain = gain
        self.initsigma=init_sigma
        self.initW =init_w
        self.trainSigmas = trainSigmas
        self.trainWeights = trainWeights
        self.trainGain = trainGain
        self.bias_initializer =init_bias
        self.bias_regularizer = reg_bias
        self.bias_constraint = None
        self.use_bias = use_bias
        self.Sigma =None
        self.sigma_regularizer = regularizers.get(sigma_regularizer)
        self.norm = norm

        #self.input_shape = input_shape
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'))
        #print(kwargs)
        #print(kernel_size,type(kernel_size))
        if type(kernel_size)==int or type(kernel_size)==float:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        print("KERNEL SIZE",self.kernel_size)
        if kernel_size ==(1,1):
            print("Kernel shape is not appropriate for Adaptive 2D Conv")
        
        self.num_filters = nfilters
        #self.incoming_channels = incoming_channels


        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(
                self.output_padding, 2, 'output_padding')
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(self.strides) + ' must be '
                                     'greater than output padding ' +str(self.output_padding))

        super(Conv2DAdaptiveDyn, self).__init__(**kwargs)


    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]

        self.input_channels = input_dim
        kernel_shape = self.kernel_size + (input_dim, self.nfilters)
        print("kernel shape:",kernel_shape)

        self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True
        # Create a trainable weight variable for this layer.

        kernel_size = self.kernel_size
        # Idxs Init
        #mu = np.array([kernel_size[0] // 2, kernel_size[1] // 2])
        mu = np.array([0.5, 0.5])


        # Convert Types
        self.mu = mu.astype(dtype='float32')

        # Shared Parameters
        # below works for only two dimensional cov
        #self.cov = self.add_weight(shape=[input_dim*self.filters,2,2],
        #                          name="cov", initializer=cov_init, trainable=False)

        #from functools import partial

        #sigma_initializer = partial(sigma_init,initsigma=self.initsigma)




        self.idxs= idx_init(shape=[kernel_size[0]*kernel_size[1],2])

        #self.Sigma = self.add_weight(shape=(self.nfilters,),
        #                                  name='Sigma',
        #                                  initializer=self.sigma_initializer,
        #                                  trainable=self.trainSigmas,
        #                                  constraint= constraints.NonNeg(),
        #                                  regularizer=self.sigma_regularizer)

        self.W = self.add_weight(shape=[kernel_size[0],kernel_size[1],
                                        self.input_channels,self.nfilters],
                                 name='Weights',
                                 initializer=initializers.he_uniform(),
                                 #initializer=initializers.orthogonal,
                                 #initializer=initializers.he_uniform(),
                                 trainable=True,
                                 regularizer = self.kernel_regularizer,
                                 constraint=None)
        print("INPUT_SHAPE-:",input_shape[1:])
        self.WSigma = self.add_weight(shape=[np.prod(input_shape[1:]),self.nfilters],
                                 name='Weights_Sigma',
                                 initializer=initializers.he_uniform(),
                                 #initializer=initializers.orthogonal,
                                 #initializer=initializers.he_uniform(),
                                 trainable=True,
                                 regularizer = self.kernel_regularizer,
                                 constraint=None)
        
        self.sigma_mean  = self.add_weight(shape=(self.nfilters,), trainable=False,
                                 regularizer = self.kernel_regularizer,
                                 initializer=initializers.ones(),
                                 name='sigma_mean',
                                 constraint=None)
        self.sigma_var  = self.add_weight(shape=(self.nfilters,), trainable=False,
                                 initializer=initializers.ones(),
                                 regularizer = self.kernel_regularizer,
                                 name='sigma_var',
                                 constraint=None)
        self.sigma_max  = self.add_weight(shape=(self.nfilters,), trainable=False,
                                 regularizer = self.kernel_regularizer,
                                 initializer=initializers.ones(),
                                 name='sigma_max',
                                 constraint=None)
        self.sigma_min  = self.add_weight(shape=(self.nfilters,), trainable=False,
                                 regularizer = self.kernel_regularizer,
                                 initializer=initializers.ones(),
                                 name='sigma_min',
                                 constraint=None) 
        

#        self.gain = self.add_weight(shape=(self.nfilters,),
#                                          name='Gain',
#                                          initializer=initializers.constant(1.0),
#                                          trainable=self.trainGain,
#                                          constraint= constraints.NonNeg(),
#                                          regularizer=None)
#                                      initializer=initializers.,
#                                      name='kernel',trainable=False,
#                                      regularizer=None,
#                                      constraint=None)
        self.kernel=None

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.nfilters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(Conv2DAdaptiveDyn, self).build(input_shape)  # Be sure to call this somewhere!



    def U(self,sigmas):

        up= K.sum((self.idxs - self.mu)**2, axis=1)
        #print("up.shape",up.shape)
        up = K.expand_dims(up,axis=1,)
        #print("up.shape",up.shape)
        # clipping scaler in range to prevent div by 0 or negative cov.
        
        sigma = K.clip(sigmas,0.001,100.0) # assuming 
        #cov_scaler = self.cov_scaler
        dwn = 2 * ( sigma ** 2)
        dwn = K.reshape(dwn,shape=(1,-1))
        #scaler = (np.pi*self.cov_scaler**2) * (self.idxs.shape[0])
        result = K.exp(-up / dwn)

        # Transpose is super important.
        #filter: A 4-D `Tensor` with the same type as `value` and shape
        #`[height, width, in_channels,output_channels]`
        # we do not care about input channels

        masks = K.reshape(result,(-1,self.kernel_size[0],
                                  self.kernel_size[1],
                                  1,self.nfilters))
        
        #print("masks shape",masks.shape)
        if self.input_channels>1:
            masks = K.repeat_elements(masks, self.input_channels, axis=3)
            #print("U masks reshaped :",masks.shape)
        #print("masks shape",masks.shape)

        #sum normalization each filter has sum 1
        #sums = K.sum(masks**2, axis=(0, 1), keepdims=True)
        #print(sums)
        #gain = K.constant(self.gain, dtype='float32')

        #Normalize to 1
        if self.norm > 0:
            masks /= K.sqrt(K.sum(K.square(masks), axis=(0, 1, 2),keepdims=True))
        if self.norm > 1:
            masks *= K.sqrt(K.constant(self.input_channels*self.kernel_size[0]*self.kernel_size[1]))
        #Normalize to get equal to WxW Filter
        #masks *= K.sqrt(K.constant(self.input_channels*self.kernel_size[0]*self.kernel_size[1]))
        # make norm sqrt(filterw x filterh x self.incoming_channel)
        # the reason for this is if you take U all ones(self.kernel_size[0],kernel_size[1], num_channels)
        # its norm will sqrt(wxhxc)
        #print("Vars: ",self.input_channels,self.kernel_size[0],self.kernel_size[1])


        return masks



    def call(self, inputs):
        input_shape = K.shape(inputs)
        batch_size = input_shape[0]
        if self.data_format == 'channels_first':
          h_axis, w_axis = 2, 3
          c_axis= 1

        else:
            h_axis, w_axis = 1, 2
            c_axis=3

        ##BTEK
        #in_channels =input_shape[c_axis]
        input_flattened = K.reshape(inputs,(input_shape[0],-1))
        #print("INPUT SHAPE:",input_flattened.shape)
        self.kernel=None
        
        self.Sigma= K.relu(K.dot(input_flattened,self.WSigma))+K.constant(0.001, dtype=np.float32)
        #K.update(self.sigma_mean, 
        #print("Here is the Self SIGMA", Sigmas.shape)
        kernel = self.U(self.Sigma)        #print(K.eval(kernel))
        # multiply with weights
        #print("Kernel shape: ", kernel.shape)
        fiw = kernel*K.expand_dims(self.W,axis=0) #*self.gain#*K.tanh(self.W)
        print("FİW shape: ", kernel.shape)
        
        def single_conv(tupl):
            x, kernel = tupl
            print("x shape", x.shape, "kernel shape", kernel.shape)
            #K.print_tensor(self.Sigmas, message="Sigma:")
            return K.conv2d(K.expand_dims(x,axis=0), kernel, strides=(1, 1), padding=self.padding)
        # Assume kernels shape is [tf.shape(inp)[0], fh, fw, c_in, c_out]
        outputs = K.squeeze(tf.map_fn(
                single_conv, (inputs, fiw), dtype=tf.float32),
            axis=1
            )
        #---------------------------------------------------------------------
        #print("Trainable weights", self._trainable_weights)
        #inputs_5d = K.expand_dims(inputs, axis=0)
        #print("INPUT 5d  shape: ", inputs_5d.shape)
        '''
        i = tf.constant(0)
        c = lambda i: tf.less(i, input_shape[0])
        outputs = K.zeros(shape=self.compute_output_shape(list(input_shape)), dtype=K.dtype(inputs))
        def body2(j, i, result):
            global similarity_matrix, U, V
            result_j = (tf.transpose(U[:, i]) * V[:, j])
            return j + 1, i, result + result_j
        r = tf.while_loop(c, bw, [i,outputs,inputs])

        outputs = K.conv2d(
                inputs,
                fiw,
                strides=(1,1),
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=None)
        
        outputs = K.squeeze(outputs, axis=0)
        print("outputs 4d  shape: ", outputs.shape)

        #print(outputs.shape)
        '''
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs



    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
        elif self.data_format == 'channels_first':
            space = input_shape[2:]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        if self.data_format == 'channels_last':
            return (input_shape[0],) + tuple(new_space) + (self.nfilters,)
        elif self.data_format == 'channels_first':
            return (input_shape[0], self.filters) + tuple(new_space)
        #return tuple(output_shape)

    def sigma_initializer(self, shape,  dtype='float32'):
        initsigma = self.initsigma

        print("Initializing sigma", type(initsigma), initsigma)

        if isinstance(initsigma,float):  #initialize it with the given scalar
            sigma = initsigma*np.ones(shape[0],dtype='float32')
        elif (isinstance(initsigma,tuple) or isinstance(initsigma,list))  and len(initsigma)==2: #linspace in range
            sigma = np.linspace(initsigma[0], initsigma[1], shape[0],dtype=dtype)
        elif isinstance(initsigma,np.ndarray) and initsigma.shape[1]==2 and shape[0]!=2: # set the values directly from array
            sigma = np.linspace(initsigma[0], initsigma[1], shape[0],dtype=dtype)
        elif isinstance(initsigma,np.ndarray): # set the values directly from array
            sigma = (initsigma).astype(dtype=dtype)
        else:
            print("Default initial sigma value 0.1 will be used")
            sigma = np.float32(0.1)*np.ones(shape[0],dtype=dtype)

        #print("Scale initializer:",sigma)
        return sigma.astype(dtype)


    def weight_initializer(self,shape, dtype='float32'):
        #only implements channel last and HE uniform
        initer = 'He'
        distribution = 'uniform'

        sigmas = K.ones(self.nfilters)*0.1
        kernel = K.eval(self.U(sigmas))
        print("Kernel max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
        W = np.zeros(shape=shape, dtype=dtype)
        print("kernel shape:", kernel.shape, ", W shape: ",W.shape)
        # for Each Gaussian initialize a new set of weights
        verbose=True
        fan_out = np.prod(self.nfilters)*self.kernel_size[0]*self.kernel_size[1]

        for c in range(W.shape[-1]):
            fan_in = np.sum((kernel[:,:,:,c])**2)

            #fan_in *= self.input_channels no need for this in repeated U.
            if initer == 'He':
                std = self.gain * sqrt32(2.0) / sqrt32(fan_in)
            else:
                std = self.gain * sqrt32(2.0) / sqrt32(fan_in+fan_out)
                # must be 1.0 like below
                #std = self.gain / sqrt32(fan_in+fan_out)

                # best working so far. U_ norm =1 , self.gain=0.5,glorot_uniform  with leaky relu

            std = np.float32(std)
            if c == 0:
                print("Std here: ",std, type(std),W.shape[:-1],
                      " fan_in", fan_in, " fan out", fan_out,"mx U", np.max(kernel[:,:,:,c]))
            if distribution == 'uniform':
                std = std * sqrt32(3.0)
                std = np.float32(std)
                w_vec = np.random.uniform(low=-std, high=std, size=W.shape[:-1])
            elif distribution == 'normal':
                std = std/ np.float32(.87962566103423978)
                w_vec = np.random.normal(scale=std, size=W.shape[:-1])

            W[:,:,:,c] = w_vec.astype('float32')

        return W
    
    
    def weight_initializer_delta_ortho(self,shape, dtype='float32'):
        #only implements channel last and HE uniform
        verbose=True
        sigmas = K.ones(self.nfilters)*0.1
        kernel = K.eval(self.U(sigmas))
        W = np.zeros(shape=shape, dtype=dtype)
        if verbose:
            print("Kernel max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
            print("kernel shape:", kernel.shape, ", W shape: ",W.shape)
            #print("num input filters must be greater than output")
            
        assert(shape[2]<=shape[3])
        
        
        # for Each Gaussian initialize a new set of weights
        
        
        ky = self.kernel_size[0]
        kx = self.kernel_size[0]
        
        a = np.random.normal(size=[shape[-1], shape[-1]])
        # Compute the qr factorization
        q, r = np.linalg.qr(a)
        #print(q.shape, r.shape)

        # Make Q uniform
        d = np.diag(r)
        q *= np.sign(d)
        q = q[:shape[-2], :]
        #print(q.shape)
        #print(q)
        #print(kernel[ky//2,kx//2,:,:])
        W[ky//2,kx//2,:,:]=q/kernel[ky//2,kx//2,:,:]
        #std = np.std(W)
        #print("Std here: ",std, type(std))
        W = W.astype('float32')
        #print("Sums here: ",np.sum(W*kernel,axis=(0,1,2)))
        #print("Max here: ",np.max(W*kernel,axis=(0,1,2)))
        #print("Min here: ",np.min(W*kernel,axis=(0,1,2)))
        #k = input()
       

        return W


def lr_schedule(epoch,lr=1e-3):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """

    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    #print('Learning rate: ', lr)
    return lr

def create_2_layer_network(input_shape,  num_classes=10, settings={}):
    from keras.models import  Model
    from keras.layers import Input, Dense, Dropout, Flatten,Conv2D, BatchNormalization
    from keras.layers import Activation, MaxPool2D
    from functools import partial
    network=[]
    network.append(Input(shape=input_shape))
    print(input_shape)
    print(settings)
    nfilters= settings['nfilters']
    afw = settings['adaptive_kernel_size']
    if settings['test_layer']=='aconv':
        Conv2F_1 = Conv2DAdaptiveDyn(rank=2,nfilters=nfilters,
                                kernel_size=(afw,afw),
                                data_format='channels_last',strides=1,
                                padding='same',name='acnn-1', activation='linear',
                                trainSigmas=True, trainWeights=True,
                                init_sigma=[0.1,0.5],
                                gain = 1.0,
                                kernel_regularizer=None,
                                sigma_regularizer=None,
                                init_bias=initializers.Constant(0),
                                norm=0,use_bias=False)
        
        Conv2F_2 = Conv2DAdaptiveDyn(rank=2,nfilters=nfilters,
                                kernel_size=(afw,afw),
                                data_format='channels_last',strides=1,
                                padding='same',name='acnn-2', activation='linear',
                                trainSigmas=True, trainWeights=True,
                                init_sigma=[0.1,0.5],
                                gain = 1.0,
                                kernel_regularizer=None,
                                init_bias=initializers.Constant(0),
                                norm=0,use_bias=False)
    elif settings['test_layer']=='conv':
        Conv2F_1 = Conv2D(filters=nfilters,
                         kernel_size=(afw,afw), padding='same',
                        activation='linear')
        Conv2F_2 = Conv2D(filters=nfilters,
                         kernel_size=(afw,afw), padding='same',
                        activation='linear')
    
    else:
        print("UNKNOWN TEST LAYER")
        return
    network.append(Conv2F_1(network[-1]))
    network.append(BatchNormalization()(network[-1]))
    network.append(Activation('relu')(network[-1]))

    network.append(Conv2F_2(network[-1]))
    network.append(BatchNormalization()(network[-1]))
    network.append(Activation('relu')(network[-1]))


    network.append(MaxPool2D((2,2))(network[-1]))
    #node_pool = MaxPool2D((4,4))(node_conv2) works good.
    network.append(Flatten(data_format='channels_last')(network[-1]))
    
    network.append(Dropout(0.5)(network[-1]))
    heu= initializers.he_uniform
  

    network.append(Dense(256,name='dense-1',activation='linear',
                          kernel_initializer=heu())(network[-1]))

    network.append(BatchNormalization()(network[-1]))
    network.append(Activation('relu')(network[-1]))
    network.append(Dropout(0.5)(network[-1]))
  

    network.append(Dense(num_classes, name='softmax', activation='softmax',
                     kernel_initializer=initializers.he_uniform(),
                     kernel_regularizer=None)(network[-1]))

    #decay_check = lambda x: x==decay_epoch

    model = keras.models.Model(inputs=[network[0]], outputs=network[-1])

    return model

def create_4_layer_network(input_shape,  num_classes=10, settings={}):
    network=[]
    network.append(Input(shape=input_shape))

    prfw = settings['pre_kernel_size']
    
    network.append(Conv2D(32, kernel_size=(prfw, prfw),
                       activation='linear',padding='same', kernel_initializer=w_ini(),
                       kernel_regularizer=None)(network[-1]))
    network.append(BatchNormalization()(network[-1]))
    network.append(Activation('relu')(network[-1]))
    #network.append(Dropout(0.2)(network[-1]))
    network.append(Conv2D(32, (prfw, prfw), activation='linear',
                     kernel_initializer=w_ini(), padding='same',
                     kernel_regularizer=None)(network[-1]))
    #odel.add(MaxPooling2D(pool_size=(2, 2)))
    network.append(BatchNormalization()(network[-1]))
    network.append(Activation('relu')(network[-1]))

    #network.append(Dropout(0.2)(network[-1]))

    #=============================================================================
    nfilters= settings['nfilters']
    #fw = settings['adaptive_kernel_size']
    afw = settings['adaptive_kernel_size']
    residual = settings['residual']
    if settings['test_layer']=='aconv':

        prev_layer = network[-1]


        conv_node = Conv2DAdaptiveDyn(rank=2,nfilters=nfilters,kernel_size=(afw,afw),
                                data_format='channels_last',strides=1,
                                padding='same',name='acnn-1', activation='linear',
                                trainSigmas=True, trainWeights=True,
                                init_sigma=[0.1,0.5],
                                gain = 1.0,
                                kernel_regularizer=None,
                                init_bias=initializers.Constant(0),
                                norm=2,
                                use_bias=False)(prev_layer)
        if residual:
            network.append(Add()([conv_node,prev_layer]))
        else:
            network.append(conv_node)
        #, input_shape=input_shape))
    elif settings['test_layer']=='conv':
        #fw = 7
        #v_ini = VS_ini(scale=0.25,mode='fan_in',distribution='uniform')
        network.append(Conv2D(nfilters, (afw, afw), activation='linear',
                         kernel_initializer=w_ini(),
                         kernel_regularizer=None,
                         padding='same')(network[-1]))
        #, input_shape=input_shape))
        
    else:
        print("UNKNOWN TEST LAYER")
        return
        
    network.append(BatchNormalization()(network[-1]))
    network.append(ReLU()(network[-1]))
    #network.append(ReLU(negative_slope=0.01)(network[-1]))
    #network.append(Activation('selu'))
    network.append(MaxPooling2D(pool_size=(2,2))(network[-1]))
    print("MAY BE MAXPOOL LAYER IS AFFECTING SIGNAL ")
    network.append(Dropout(0.2)(network[-1]))
    #model.add(keras.layers.AlphaDropout(0.2))
    #network.append(GlobalAveragePooling2D()(network[-1]))
    network.append(Flatten()(network[-1]))
    network.append(Dense(units=128, activation='linear',
                   kernel_regularizer=None)(network[-1]))
    network.append(BatchNormalization()(network[-1]))
    network.append(ReLU()(network[-1]))
    network.append(Dropout(0.2)(network[-1]))


    network.append(Dense(num_classes, activation='softmax',
                    kernel_regularizer=None)(network[-1]))

    model = keras.models.Model(inputs=[network[0]], outputs=network[-1])

    return model



def test_mnist(settings,sid=9):
    

    #sess = K.get_session()
    K.clear_session()
    #K.set_session(sess)
    np.random.seed(sid)
    tf.random.set_random_seed(sid)
    tf.compat.v1.random.set_random_seed(sid)


    #dset='cifar10'

    dset = settings['dset']
    batch_size = settings['batch']
    num_classes = 10
    epochs =settings['epochs']
    test_acnn = settings['test_layer']=='aconv'


    if dset=='mnist':
        # input image dimensions
        img_rows, img_cols = 28, 28
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        n_channels=1

    elif dset=='cifar10':
        img_rows, img_cols = 32,32
        n_channels=3

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
    elif settings['dset']=='fashion':
        img_rows, img_cols = 28,28
        n_channels=1

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        
    elif settings['dset']=='mnist-clut':
        
        img_rows, img_cols = 60, 60  
        # the data, split between train and test sets
        
        folder='/media/home/rdata/image/'
        #folder='/home/btek/datasets/image/'
        data = np.load(folder+"mnist_cluttered_60x60_6distortions.npz")
    
        x_train, y_train = data['x_train'], np.argmax(data['y_train'],axis=-1)
        x_valid, y_valid = data['x_valid'], np.argmax(data['y_valid'],axis=-1)
        x_test, y_test = data['x_test'], np.argmax(data['y_test'],axis=-1)
        x_train=np.vstack((x_train,x_valid))
        y_train=np.concatenate((y_train, y_valid))
        n_channels=1
        
        #decay_epochs =[e_i*30,e_i*100]
            
    elif settings['dset']=='lfw_faces':
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
        n_classes = target_names.shape[0]
        
        print("Total dataset size:")
        print("n_samples: %d" % n_samples)
        print("n_features: %d" % n_features)
        print("n_classes: %d" % n_classes)
               

        

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], n_channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], n_channels, img_rows, img_cols)
        input_shape = (n_channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, n_channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, n_channels)
        input_shape = (img_rows, img_cols, n_channels)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    trn_mn = np.mean(x_train, axis=0)
    x_train -= trn_mn
    x_test -= trn_mn
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if settings['arch']=='simple':
        model = create_2_layer_network(input_shape, num_classes, settings)
    else:
        model = create_4_layer_network(input_shape, num_classes, settings)
        
    model.summary()


    # from lr_multiplier import LearningRateMultiplier
    #lr=0.001
    #multipliers = {'acnn-1/Sigma:0': 1.0,'acnn-1/Weights:0': 1000.0,
    #               'acnn-2/Sigma:0': 1.0,'acnn-2/Weights:0': 1000.0}
    #opt = LearningRateMultiplier(SGD, lr_multipliers=multipliers,
    #                             lr=lr, momentum=0.9,decay=0)

    #opt= SGD(lr=lr,momentum=0.9,decay=0,nesterov=False)
    '''lr_dict = {'all':0.01,'acnn-1/Sigma:0': 0.01,'acnn-1/Weights:0': 1.0,
                   'acnn-2/Sigma:0': 0.01,'acnn-2/Weights:0': 0.1}


    mom_dict = {'all':0.9,'acnn-1/Sigma:0': 0.5,'acnn-1/Weights:0': 0.9,
                   'acnn-2/Sigma:0': 0.9,'acnn-2/Weights:0': 0.9}


    decay_dict = {'all':0.95, 'acnn-1/Sigma:0': 0.05, 'acnn-1/Weights:0':0.95,
                  'acnn-1/Sigma:0': 0.05,'acnn-2/Weights:0': 0.95}

    clip_dict = {'acnn-1/Sigma:0':(0.05,1.0),'acnn-2/Sigma:0':(0.05,1.0)}
    '''

    #
    #           'acnn-2/Sigma:0': 0.00001,'acnn-2/Weights:0': 1.0}
    lr_dict = {'all':0.1}
    for i in lr_dict.keys(): 
        lr_dict[i]*=settings['lr_multiplier']
        
    mom_dict = {'all':0.9}
    clip_dict ={}# {'Sigma': [0.05, 2.0]}
    decay_dict = {'all':0.1}
    e_i = x_train.shape[0] // batch_size

    decay_epochs =np.array([e_i*20,e_i*80], dtype='int64')
    #print("WHAT THE ", lr_dict)
    opt = SGDwithLR(lr=lr_dict, momentum = mom_dict, decay=decay_dict,
                    clips=clip_dict,decay_epochs=decay_epochs, 
                    verbose=2, update_clip=1.0)
    
    
    #opt = AdamwithClip(clips=clip_dict,clipvalue=1.0)
    


#    chkpt= keras.callbacks.ModelCheckpoint('best-model.h5',
#                                    monitor='val_acc',
#                                    verbose=1,
#                                    save_best_only=True,
#                                    save_weights_only=True,
#                                    mode='max', period=1)


#    tb = TensorBoard(log_dir='./tb_logs/mnist/acnn-res-lr5',
#                     histogram_freq = 1,
#                     write_grads=True,
#                     write_graph=False)
    #callbacks = [tb]
    
    stat_func_name = ['max: ', 'mean: ', 'min: ', 'var: ', 'std: ']
    stat_func_list = [np.max, np.mean, np.min, np.var, np.std]
    
    callbacks = []
    
    
    if test_acnn:
        pr_1 = PrintLayerVariableStats("acnn-1","Weights:0",stat_func_list,stat_func_name)
        pr_2 = PrintLayerVariableStats("acnn-1","Sigma:0",stat_func_list,stat_func_name)
        pr_3 = PrintLayerVariableStats("acnn-1","bias:0",stat_func_list,stat_func_name)
        #rv_weights_1 = RecordVariable("acnn-1","Weights:0")
        #rv_sigma_1 = RecordVariable("acnn-1","Sigma:0")
        pr_sig = PrintAnyOutputVariable(model,model.get_layer("acnn-1").Sigma,
                                        stat_func_list, stat_func_name,
                                        x_test)
        

        callbacks+=[pr_1, pr_2, pr_3,pr_sig] #,rv_weights_1,rv_sigma_1]
    else:
        #pr_1 = PrintLayerVariableStats("conv2d_3","kernel:0",stat_func_list,stat_func_name)
        #rv_weights_1 = RecordVariable("conv2d_3","kernel:0")
        pass
        #callbacks+=[pr_1, rv_weights_1]
        #pr_3 = PrintLayerVariableStats("conv2d_1","kernel:0",stat_func_list,stat_func_name)
        #rv_kernel = RecordVariable("conv2d_1","kernel:0")
        #callbacks+=[pr_3,rv_kernel]
    pr_4 = PrintLayerVariableStats("batch_normalization_1","moving_mean:0",
                                       stat_func_list,stat_func_name,not_trainable=True)
    pr_5 = PrintLayerVariableStats("batch_normalization_1",
                                       "moving_variance:0",
                                       stat_func_list,
                                       stat_func_name,not_trainable=True)
    callbacks+=[pr_4,pr_5] 
    print("CALLBACKS:",callbacks)
    
    #print("TRAINABLE WEIGHTS:",model.trainable_weights)

    #print("WARNING by BTEK: if you see An operation has `None` for gradient. \
    #      Please make sure that all of your ops have a gradient defined \
    #      (i.e. are differentiable). Common ops without gradient: \
    #          K.argmax, K.round, K.eval. REMOVE TENSORBOARD CALLBACK OR EDIT IT!")


    #print(opt)
    #opt = SGD(lr=0.01,momentum=0.9)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])

    plot_this = False
    if plot_this and test_acnn:
        print("Plotting kernels before...")
        import matplotlib.pyplot as plt
        acnn_layer = model.get_layer('acnn-1')
        ws = acnn_layer.get_weights()
        print("Sigmas before",ws[0])
        u_func = K.function(inputs=[model.input], outputs=[acnn_layer.U()])
        output_func = K.function(inputs=[model.input], outputs=[acnn_layer.output])

        U_val=u_func([np.expand_dims(x_test[0], axis=0)])

        print("U shape", U_val[0].shape)
        print("U max:", np.max(U_val[0][:,:,:,:]))
        num_filt=min(U_val[0].shape[3],12)
        fig=plt.figure(figsize=(20,8))
        for i in range(num_filt):
            ax1=plt.subplot(1, num_filt, i+1)
            im = ax1.imshow(np.squeeze(U_val[0][:,:,0,i]))
        fig.colorbar(im, ax=ax1)

        plt.show(block=False)

        fig=plt.figure(figsize=(20,8))
        num_show = min(U_val[0].shape[3],12)
        indices = np.int32(np.linspace(0,U_val[0].shape[3]-1,num_show))
        for i in range(num_show):
            ax1=plt.subplot(1, num_filt, i+1)
            #print("U -shape: ", acnn_layer.U().shape,type(K.eval(acnn_layer.U()[:,:,0,i])))
            #print("Prod-shape", (ws[1][:,:,0,i]*acnn_layer.U()[:,:,0,i]).shape)
            plt.imshow(np.float32(ws[1][:,:,0,indices[i]]*
                                  K.eval(acnn_layer.U()[:,:,0,indices[i]])))

        plt.show(block=False)

    # Run training, with or without data augmentation.
    data_augmentation = settings['data_augmentation']
    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(x_train, y_train,
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
            rotation_range=5,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.0,
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
            data_format='channels_last',
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks,
                            steps_per_epoch=x_train.shape[0]//batch_size)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    

    plot_this=False
    if plot_this and test_acnn:
        print("Plotting kernels after ...")

        print("U max:", np.max(U_val[0][:,:,:,:]))
        import matplotlib.pyplot as plt
        ws = acnn_layer.get_weights()
        print("Sigmas after",ws[0])
        U_val=u_func([np.expand_dims(x_test[2], axis=0)])

        print("U shape", U_val[0].shape)
        num_filt=min(U_val[0].shape[3],12)

        indices = np.int32(np.linspace(0,U_val[0].shape[3]-1,num_filt))

        fig=plt.figure(figsize=(16,5))
        for i in range(num_filt):
            ax=plt.subplot(1, num_filt, i+1)
            kernel_u = U_val[0][:,:,0,indices[i]]
            im = ax.imshow(np.squeeze(kernel_u))
            print("kernel mean,var,max,min",np.mean(kernel_u),
                                           np.var(kernel_u),
                                           np.max(kernel_u), np.min(kernel_u))
        #fig.colorbar(im, ax=ax1)
        plt.show(block=False)


        print("outputs  ...")

        n=5

        out_val=output_func([np.expand_dims(x_test[5], axis=0)])
        print("Outputs shape", out_val[0].shape)
        num_filt=min(out_val[0].shape[3],12)

        indices = np.int32(np.linspace(0,out_val[0].shape[3]-1,num_filt))
        fig=plt.figure(figsize=(20,8))
        ax=plt.subplot(1, num_filt+1, 1)
        im = ax.imshow(np.squeeze(x_test[5]))
        print(y_test[5])
        print("input mean,var,max",np.mean(x_test[n]),np.var(x_test[n]),np.max(x_test[n]))
        for i in range(num_filt):
            ax=plt.subplot(1, num_filt+1, i+2)
            out_im = out_val[0][0,:,:,indices[i]]
            im = ax.imshow(np.squeeze(out_im))

            print("ouput mean,var,max",np.mean(out_im),
                                           np.var(out_im),
                                           np.max(out_im),np.min(out_im))
            #plt.colorbar(im,ax=ax)
        plt.show(block=False)

        print("Weights")
        fig=plt.figure(figsize=(20,8))
        num_show = min(U_val[0].shape[3],12)
        indices = np.int32(np.linspace(0,U_val[0].shape[3]-1,num_show))
        for i in range(num_show):
            ax1=plt.subplot(1, num_filt, i+1)
            #print("U -shape: ", acnn_layer.U().shape,type(K.eval(acnn_layer.U()[:,:,0,i])))
            #print("Prod-shape", (ws[1][:,:,0,i]*acnn_layer.U()[:,:,0,i]).shape)
            plt.imshow(np.float32(ws[1][:,:,0,indices[i]]),cmap='gray')

        plt.show(block=False)

        print("ACNN Filters after")
        fig=plt.figure(figsize=(20,8))
        num_show = min(U_val[0].shape[3],12)
        indices = np.int32(np.linspace(0,U_val[0].shape[3]-1,num_show))
        for i in range(num_show):
            ax1=plt.subplot(1, num_filt, i+1)
            #print("U -shape: ", acnn_layer.U().shape,type(K.eval(acnn_layer.U()[:,:,0,i])))
            #print("Prod-shape", (ws[1][:,:,0,i]*acnn_layer.U()[:,:,0,i]).shape)
            plt.imshow(np.float32(ws[1][:,:,0,indices[i]]*
                                  K.eval(acnn_layer.U()[:,:,0,indices[i]])),cmap='gray')

        plt.show(block=False)


        cnn_layer = model.get_layer('conv2d_1')
        wcnn = cnn_layer.get_weights()
        print("CNN Filters of", cnn_layer)
        fig=plt.figure(figsize=(20,8))
        num_show = min(wcnn[0].shape[3],12)
        indices = np.int32(np.linspace(0,wcnn[0].shape[3]-1,num_show))
        for i in range(num_show):
            ax1=plt.subplot(1, num_filt, i+1)
            #print("U -shape: ", acnn_layer.U().shape,type(K.eval(acnn_layer.U()[:,:,0,i])))
            #print("Prod-shape", (ws[1][:,:,0,i]*acnn_layer.U()[:,:,0,i]).shape)
            plt.imshow(np.float32(wcnn[0][:,:,0,indices[i]]),cmap='gray')

        plt.show(block=False)


        rv_sigma_arr = np.array(rv_sigma_1.record)
        fig=plt.figure(figsize=(4,8))
        plt.plot(rv_sigma_arr)
        plt.title('Sigma')
        plt.show(block=False)

        rv_weights_arr = np.array(rv_weights_1.record)
        rv_weights_arr2d = np.reshape(rv_weights_arr,
                            (rv_weights_arr.shape[0],
                             np.prod(rv_weights_arr.shape[1:])))
        print(rv_weights_arr.shape)
        fig=plt.figure(figsize=(4,8))
        klist=[1,1,5,9,12,15,18,25,32,132,1132]
        for i in klist:
            plt.plot(rv_weights_arr2d[:,i])
        plt.title('weights-acnn')
        plt.show(block=False)



        rv_kernel_arr = np.array(rv_kernel.record)
        rv_kernel_arr2d = np.reshape(rv_kernel_arr,
                            (rv_kernel_arr.shape[0],
                             np.prod(rv_kernel_arr.shape[1:])))
        print(rv_kernel_arr.shape)
        fig=plt.figure(figsize=(4,8))
        klist=[1,1,5,9,12,15,18,25,32]
        for i in klist:
            plt.plot(rv_kernel_arr2d[:,i])
        plt.title('weights-conv2d-1')
        plt.show(block=False)
        
    return score, history, model, callbacks



#tf.reset_default_graph()

def repeated_trials(settings):
    list_scores =[]
    list_histories =[]
    for i in range(settings['repeats']):
        print("--------------------------------------------------------------")
        print("REPEAT:",i)
        print("--------------------------------------------------------------")
        sc, hs, ms, cb = test_mnist(settings,sid=31+i*17)
        list_scores.append(sc)
        list_histories.append(hs)
        
    print("Final scores", list_scores)
    mx_scores = [np.max(list_histories[i].history['val_acc']) for i in range(len(list_histories))]
    histories = [h.history for h in list_histories]
    print("Max sscores", mx_scores)
    print("Max stats", np.mean(mx_scores), np.std(mx_scores))
    return mx_scores, histories, list_scores




if __name__ == '__main__':
    #import test_filters
    #exit()
    import sys
    kwargs = {'dset':'cifar10', 'arch':'simple', 'repeats':1, 
              'test_layer':'aconv',
              'epochs':50, 'batch':256, 'exname':'noname', 
              'adaptive_kernel_size':7, 'nfilters':16, 
              'data_augmentation':False, 'lr_multiplier':0.1}
    # For MNIST YOU CAN USE lr_multiplier 1.0, 
    # FOR OTHER USE lr_multiplier 0.1 or lower
    if len(sys.argv) > 1:
            kwargs['dset'] = sys.argv[1]
    if len(sys.argv) > 2:
        kwargs['arch'] = sys.argv[2]
    if len(sys.argv) > 3:
        kwargs['repeats'] = int(sys.argv[3])
        print("repeating ", kwargs['repeats'], " reps")
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
        kwargs['exname'] = sys.argv[7]
        print("exname", sys.argv[7])
    if len(sys.argv) > 8:
        kwargs['adaptive_kernel_size'] = int(sys.argv[8])
        print("adaptive_kernel_size", int(sys.argv[8]))
    if len(sys.argv) > 9:
        kwargs['nfilters'] = int(sys.argv[9])
        print("nfilters", int(sys.argv[9]))
    if len(sys.argv) > 10:
        kwargs['data_augmentation'] = sys.argv[10]=="True"
        print("data_augmentation", sys.argv[10]=="True")
    if len(sys.argv) > 11:
        kwargs['lr_multiplier'] = float(sys.argv[11])
        print("lr_multiplier", float(sys.argv[11]))

    r = repeated_trials(kwargs)
    #print(r)
    
    import time 
    delayed_start = 0 
    print("Delayed start ",delayed_start)
    time.sleep(delayed_start)
    from datetime import datetime
    now = datetime.now()
    timestr = now.strftime("%Y%m%d-%H%M%S")
    ks = str(kwargs['adaptive_kernel_size'])+'x'+str(kwargs['adaptive_kernel_size'])
    filename = 'outputs/'+kwargs['exname']+'_'+kwargs['arch']+'_'+kwargs['dset']+'_'+kwargs['test_layer']+'_'+ks+'_'+timestr+'_'+'._results.npz'
    np.savez_compressed(filename,mod=kwargs,mx_scores =r[0], results=r)

#Max sscores [0.9875999994277954, 0.9858999995231629, 0.9874999995231628, 0.9861999994277955, 0.9858999995231629]
#Max stats 0.986619999485016 0.0007678541389076846

#u = np.load('simple_mnist_aconv_20191129-224704_._results.npz')