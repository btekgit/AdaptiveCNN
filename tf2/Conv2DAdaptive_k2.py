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
if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
    os.environ['CUDA_VISIBLE_DEVICES']="0"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import activations, regularizers, constraints, Model
from tensorflow.keras import initializers
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Add, ReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization,Activation, Concatenate
from tensorflow.keras.regularizers import l2,l1
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adadelta
#from keras.initializers import glorot_uniform as w_ini
from tensorflow.keras.initializers import he_uniform as w_ini
from tensorflow.keras.initializers import VarianceScaling as VS_ini
from tensorflow.keras import backend as K
from keras_utils_tf2 import RecordVariable, \
PrintLayerVariableStats, SGDwithLR, AdamwithClip, ClipCallback, RenormCallback
from keras_data_tf2 import load_dataset

from tensorflow.python.framework import dtypes


from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
# pylint: enable=unused-import
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn


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
    return cov



class Conv2DAdaptive(Layer):
    def __init__(self,  filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=False,
                 gain=1.0,
                 init_sigma=0.1,
                 trainSigmas=True,
                 trainWeights=True,
                 trainGain=False,
                 reg_bias=None,
                 norm =2,
                 sigma_regularizer=None,
                 kernel_initializer='he_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        super(Conv2DAdaptive, self).__init__(**kwargs)
        #def __init__(self, num_filters, kernel_sigmaze, incoming_channels=1, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activations.get(activation)
        self.gain = gain
        self.initsigma=init_sigma
        self.trainSigmas = trainSigmas
        self.trainWeights = trainWeights
        self.trainGain = trainGain
        self.norm = norm
        self.use_bias = use_bias
        self.sigma_regularizer = regularizers.get(sigma_regularizer)
        
        self.kernel_initializer=initializers.get(kernel_initializer)
        self.kernel_regularizer=regularizers.get(kernel_regularizer)
        self.bias_initializer=initializers.get(bias_initializer)
        self.bias_regularizer=regularizers.get(bias_regularizer)
        self.activity_regularizer=regularizers.get(activity_regularizer)
        self.kernel_constraint=constraints.get(kernel_constraint)
        self.bias_constraint=constraints.get(bias_constraint)

        
        #self.input_shape = input_shape
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'))
        #print(kwargs)
        #print(kernel_size,type(kernel_size))
        if type(kernel_size)==int or type(kernel_size)==float:
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if kernel_size ==(1,1):
            print("Kernel shape is not appropriate for Adaptive 2D Conv")

        self.num_filters = filters
        #self.incoming_channels = incoming_channels


        self.output_padding = output_padding
        if self.output_padding is not None:
            #self.output_padding = self.output_padding, 2, 'output_padding')
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(self.strides) + ' must be '
                                     'greater than output padding ' +str(self.output_padding))

        super(Conv2DAdaptive, self).__init__(**kwargs)


    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.input_channels = input_dim
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        print("kernel shape:",kernel_shape)

        self.bias = None
        # Set input spec.

        self.built = True
        # Create a trainable weight variable for this layer.

        kernel_size = self.kernel_size

        mu = np.array([0.5, 0.5])

        # Convert Types
        self.mu = mu.astype(dtype='float32')
        #from functools import partial


        self.idxs= idx_init(shape=[kernel_size[0]*kernel_size[1],2])
        
        s_init = self.sigma_initializer#((self.filters,),dtype='float32')
        print(self.sigma_regularizer)
        self.Sigma = self.add_weight(shape=(self.filters,),
                                          name='Sigma',
                                          initializer=s_init,
                                          trainable=self.trainSigmas,
                                          constraint= None,
                                          regularizer=self.sigma_regularizer,
                                          dtype='float32')
    
        self.W = self.add_weight(shape=[kernel_size[0],kernel_size[1],
                                        self.input_channels, self.filters],
                                 name='Weights',
                                 initializer=self.kernel_initializer,
                                 trainable=True,
                                 regularizer = self.kernel_regularizer,
                                 constraint=None)
        self.kernel=None

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(Conv2DAdaptive, self).build(input_shape)  # Be sure to call this somewhere!

    @tf.function
    def calc_U(self):

        up= K.sum((self.idxs - self.mu)**2, axis=1)
        #print("up.shape",up.shape)
        up = K.expand_dims(up,axis=1,)
        # print("up.shape",up.shape)
        # I made MIN_SI function of kernel_size however, it must be put to optimizers
        # MIN_SI MUST BE ~ 1.0/(kernel_size[0])
        # MIN_SI = 1.0/(self.kernel_size[0]*self.kernel_size[1])
        # sigma = K.clip(self.Sigma, MIN_SI, 100.0)  not necessary anymore aug 2020, because of higher clips and 
        # cov_scaler = self.cov_scaler
        dwn = 2 * ( self.Sigma ** 2)+1e-8 # 1e-8 is to prevent div by zero
        #scaler = (np.pi*self.cov_scaler**2) * (self.idxs.shape[0])
        result = K.exp(-up / dwn)

        # we do not care about input channels, later it will be broadcasted to input size. 

        masks = K.reshape(result,(self.kernel_size[0],
                                  self.kernel_size[1],
                                  1,self.filters))
        
        if self.input_channels>1:
            masks = K.repeat_elements(masks, self.input_channels, axis=2)
        

       #Normalize to 1
        if self.norm == 1:
            masks /= K.sqrt(K.sum(K.square(masks), axis=(0, 1, 2),keepdims=True))
        elif self.norm == 2:
            masks /= K.sqrt(K.sum(K.square(masks), axis=(0, 1, 2),keepdims=True))
            masks *= K.sqrt(K.constant(self.input_channels*self.kernel_size[0]*self.kernel_size[1]))
            #masks *= K.sqrt(K.constant(self.kernel_size[0]*self.kernel_size[1]))
       
        # @tf.custom_gradient
        # def threshold(x):
        #     leak = K.constant(1e-1, 'float32')
        #     y  = K.cast(x > 0.5,'float32') + leak
        #     #y = K.random_(shape=x.shape,p=x, dtype=x.dtype)
        #     def grad(dy):
        #         return dy#K.cast(dy*1.0, 'float32')
        #     return y, grad
        
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
        #print("Calling self.U:")
        self.kernel=None
        kernel = self.calc_U()        
        # 
        fiw = kernel*self.W
 

        outputs = K.conv2d(
                inputs,
                fiw,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        #print(outputs.shape)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
            
        if self.activation is not None:
            outputs = self.activation(outputs)
            
        norm_inout=True
        if norm_inout:
            instd   = K.std(inputs, axis=[1,2,3], keepdims=True)
            outstd  = K.std(outputs, axis=[1,2,3], keepdims=True)
            outputs /= outstd
            outputs *= instd
            
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
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        elif self.data_format == 'channels_first':
            return (input_shape[0], self.filters) + tuple(new_space)
        #return tuple(output_shape)

    def sigma_initializer(self, shape,  dtype='float32'):
        initsigma = self.initsigma
       
        print("Initializing sigma", type(initsigma), initsigma, type(dtype))

        if isinstance(initsigma,float):  #initialize it with the given scalar
            sigma = initsigma*np.ones(shape[0])
        elif (isinstance(initsigma,tuple) or isinstance(initsigma,list))  and len(initsigma)==2: #linspace in range
            sigma = np.linspace(initsigma[0], initsigma[1], shape[0])
        elif isinstance(initsigma,np.ndarray) and initsigma.shape[1]==2 and shape[0]!=2: # set the values directly from array
            sigma = np.linspace(initsigma[0], initsigma[1], shape[0])
        elif isinstance(initsigma,np.ndarray): # set the values directly from array
            sigma = np.convert_to_tensor(initsigma)
        else:
            print("Default initial sigma value 0.1 will be used")
            sigma = np.float32(0.1)*np.ones(shape[0])

        #print("Scale initializer:",sigma)
        return sigma.astype(dtype.as_numpy_dtype)


    def weight_initializer(self,shape, dtype='float32'):
        #only implements channel last and HE uniform
        initer = 'He'
        distribution = 'uniform'

        kernel = K.eval(self.calc_U())
        print("Kernel max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
        W = np.zeros(shape=shape, dtype=dtype.as_numpy_dtype)
        print("kernel shape:", kernel.shape, ", W shape: ",W.shape)
        # for Each Gaussian initialize a new set of weights
        verbose=True
        fan_out = np.prod(self.filters)*self.kernel_size[0]*self.kernel_size[1]

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
        
        
        W = W/np.sqrt(np.sum(W**2, axis=(0,1,2), keepdims=True))*np.sqrt(2.0)


        return W
    
    
    def weight_initializer_delta_ortho(self,shape, dtype='float32'):
        #only implements channel last and HE uniform
        verbose=True
        kernel = K.eval(self.U())
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
        W = W.astype('float3set_seed2')
        #print("Sums here: ",np.sum(W*kernel,axis=(0,1,2)))
        #print("Max here: ",np.max(W*kernel,axis=(0,1,2)))
        #print("Min here: ",np.min(W*kernel,axis=(0,1,2)))
        #k = input()
       

        return W
    
    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'norm': self.norm,
            'gain':self.gain,
            'initsigma':self.initsigma,
            'trainSigmas':self.trainSigmas,
            'trainWeights':self.trainWeights,          
            #'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer)
        }
        
        base_config = super(Conv2DAdaptive, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    

class Conv2DTransposeAdaptive(Conv2DAdaptive):
  print("class not ready, must be tested!")
  pass
  """Transposed convolution layer (sometimes called Deconvolution).

  The need for transposed convolutions generally arises
  from the desire to use a transformation going in the opposite direction
  of a normal convolution, i.e., from something that has the shape of the
  output of some convolution to something that has the shape of its input
  while maintaining a connectivity pattern that is compatible with
  said convolution.

  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers, does not include the sample axis),
  e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
  in `data_format="channels_last"`.

  Arguments:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive)20618445, shape=(), dtype=float32) tf.Tensor(0.20155334, shape=(), dtype=float32) tf.Tensor(0.2, shape=(), dtype=float32)
Max U: tf.Tensor(28.091219, shape=(), dtype=float32) Min u: tf.Tensor(0.05465.
    output_padding: An integer or tuple/list of 2 integers,
      specifying the amount of padding along the height and width
      of the output tensor.
      Can be a single integer to specify the same value for all
      spatial dimensions.
      The amount of output padding along a given dimension must be
      lower than the stride along that same dimension.
      If set to `None` (default), the output shape is inferred.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    dilation_rate: an integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix (
      see `keras.initializers`).
    bias_initializer: Initializer for the bias vector (
      see `keras.initializers`).
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation") (see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`).

  Input shape:
    4D tensor with shape:
    `(batch_size, channels, rows, cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(batch_size, rows, cols, channels)` if data_format='channels_last'.

  Output shape:
    4D tensor with shape:
    `(batch_size, filters, new_rows, new_cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(batch_size, new_rows, new_cols, filters)` if data_format='channels_last'.
    `rows` and `cols` values might have changed due to padding.
    If `output_padding` is specified:
    ```
    new_rows = ((rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0] +
    output_padding[0])
    new_cols = ((cols - 1) * strides[1] + kernel_size[1] - 2 * padding[1] +
    output_padding[1])
    ```

  Returns:
    A tensor of rank 4 representing
    `activation(conv2dtranspose(inputs, kernel) + bias)`.

  Raises:
    ValueError: if `padding` is "causal".
    ValueError: when both `strides` > 1 and `dilation_rate` > 1.

  References:
    - [A guide to convolution arithmetic for deep
      learning](https://arxiv.org/abs/1603.07285v1)
    - [Deconvolutional
      Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               output_padding=None,
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               init_sigma=0.1,
               trainSigmas=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               sigma_regularizer=None,
               norm=2,
               **kwargs):
    super(Conv2DTransposeAdaptive, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activations.get(activation),
        use_bias=use_bias,
        init_sigma = init_sigma,
        trainSigmas = trainSigmas,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        sigma_regularizer = regularizers.get(sigma_regularizer),
        norm=norm,
        **kwargs)

    self.output_padding = output_padding
    if self.output_padding is not None:
      self.output_padding = conv_utils.normalize_tuple(
          self.output_padding, 2, 'output_padding')
      for stride, out_pad in zip(self.strides, self.output_padding):
        if out_pad >= stride:
          raise ValueError('Stride ' + str(self.strides) + ' must be '
                           'greater than output padding ' +
                           str(self.output_padding))

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if len(input_shape) != 4:
      raise ValueError('Inputs should have rank 4. Received input shape: ' +
                       str(input_shape))
    if self.data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
        # it may be none 
        self.data_format ='channels_last'
    
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
    kernel_shape = self.kernel_size + (self.filters, input_dim)
    self.u_shape = kernel_shape
    
    mu = np.array([0.5, 0.5])


        # Convert Types
    self.mu = mu.astype(dtype='float32')
        #from functools import partial

        #sigma_initializer = partial(sigma_init,initsigma=self.initsigma)

    self.idxs= idx_init(shape=[kernel_shape[0]*kernel_shape[1],2])
    s_init = self.sigma_initializer#((self.filters,),dtype='float32')
    #print(self.sigma_regularizer)
    #input("eaitawre")
    self.Sigma = self.add_weight(shape=(self.filters,),
                                          name='Sigma',
                                          initializer=s_init,
                                          trainable=self.trainSigmas,
                                          constraint= None,
                                          regularizer=self.sigma_regularizer,
                                          dtype=self.dtype)
    
    self.kernel=None
    self.W = self.add_weight(
        name='Weight',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
        self.bias = self.add_weight(name='bias',
                                    shape=(self.filters,),
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    trainable=True,
                                    dtype=self.dtype)
    else:
        self.bias = None
   
    self.built = True

  #@tf.function
  def calc_UT(self):
      
    u_shape = self.u_shape

    up= K.sum((self.idxs - self.mu)**2, axis=1)
      #print("up.shape",up.shape)
    up = K.expand_dims(up,axis=1,)
      #print("up.shape",up.shape)
      # I made MIN_SI function of kernel_size however, it must be put to optimizers
      # MIN_SI MUST BE ~ 1.0/(kernel_size[0])
    MIN_SI = 1.0/(self.kernel_size[0]*self.kernel_size[1])
    sigma = K.clip(self.Sigma,MIN_SI,100.0) 
      #cov_scaler = self.cov_scaler
    dwn = 2 * ( self.Sigma ** 2)+1e-4 # 1e-4 is to prevent div by zero
      #scaler = (np.pi*self.cov_scaler**2) * (self.idxs.shape[0])
    result = K.exp(-up / dwn)
    
    
    masks = K.reshape(result,(u_shape[0:3]+(1,)))
    masks =  K.repeat_elements(masks, u_shape[3], axis=-1)
    #print("Mask shape", masks.shape)

    #Normalize to 1
    if self.norm == 1:
      masks /= K.sqrt(K.sum(K.square(masks), axis=(0, 1, 3),keepdims=True))
    elif self.norm == 2:
      masks /= K.sqrt(K.sum(K.square(masks), axis=(0, 1, 3),keepdims=True))
      
      masks *= K.sqrt(K.constant(u_shape[3]*self.kernel_size[0]*self.kernel_size[1]))
      #print(K.sqrt(K.sum(K.square(masks), axis=(0, 1, 3))))
    return masks
    
    
  def call(self, inputs):
    inputs_shape = array_ops.shape(inputs)
    batch_size = inputs_shape[0]
    if self.data_format == 'channels_first':
      h_axis, w_axis = 2, 3
    else:
      h_axis, w_axis = 1, 2

    height, width = inputs_shape[h_axis], inputs_shape[w_axis]
    kernel_h, kernel_w = self.kernel_size
    stride_h, stride_w = self.strides
    
    

    if self.output_padding is None:
      out_pad_h = out_pad_w = None
    else:
      out_pad_h, out_pad_w = self.output_padding

    # Infer the dynamic output shape:
    #print("Kernel size is \n\n", height, kernel_h, self.padding, out_pad_h, 
    #      stride_h, self.dilation_rate[0], self.data_format)
    out_height = conv_utils.deconv_output_length(height,
                                                 kernel_h,
                                                 padding=self.padding,
                                                 output_padding=out_pad_h,
                                                 stride=stride_h,
                                                 dilation=self.dilation_rate[0])
    out_width = conv_utils.deconv_output_length(width,
                                                kernel_w,
                                                padding=self.padding,
                                                output_padding=out_pad_w,
                                                stride=stride_w,
                                                dilation=self.dilation_rate[1])
    if self.data_format == 'channels_first':
      output_shape = (batch_size, self.filters, out_height, out_width)
    else:
      output_shape = (batch_size, out_height, out_width, self.filters)

    output_shape_tensor = array_ops.stack(output_shape)
    
    self.kernel=None
    #print("\nSigma:", K.max(self.Sigma), K.mean(self.Sigma), K.min(self.Sigma))
    kernel = self.calc_UT()
    #print("Max U:",K.max(kernel),"Min u:", K.min(kernel))
    #print("KERNEL_SHAPE:", kernel.shape)
    fiw = kernel*self.W
    outputs = K.conv2d_transpose(
        inputs,
        fiw,
        output_shape_tensor,
        strides=self.strides,
        padding=self.padding,
        data_format=self.data_format,
        dilation_rate=self.dilation_rate)

    if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape(inputs.shape)
      outputs.set_shape(out_shape)

    
    if self.use_bias:
      outputs = nn.bias_add(
          outputs,
          self.bias,
          data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    output_shape = list(input_shape)
    if self.data_format == 'channels_first':
      c_axis, h_axis, w_axis = 1, 2, 3
    else:
      c_axis, h_axis, w_axis = 3, 1, 2

    kernel_h, kernel_w = self.kernel_size
    stride_h, stride_w = self.strides

    if self.output_padding is None:
      out_pad_h = out_pad_w = None
    else:
      out_pad_h, out_pad_w = self.output_padding

    output_shape[c_axis] = self.filters
    output_shape[h_axis] = conv_utils.deconv_output_length(
        output_shape[h_axis],
        kernel_h,
        padding=self.padding,
        output_padding=out_pad_h,
        stride=stride_h,
        dilation=self.dilation_rate[0])
    output_shape[w_axis] = conv_utils.deconv_output_length(
        output_shape[w_axis],
        kernel_w,
        padding=self.padding,
        output_padding=out_pad_w,
        stride=stride_w,
        dilation=self.dilation_rate[1])
    return tensor_shape.TensorShape(output_shape)

  def get_config(self):
    config = super(Conv2DTransposeAdaptive, self).get_config()
    config['output_padding'] = self.output_padding
    return config

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
    from functools import partial
    network=[]
    network.append(Input(shape=input_shape))
    print(input_shape)
    print(settings)
    nfilters= settings['nfilters']
    afw = settings['adaptive_kernel_size']
    if settings['test_layer']=='aconv':
        Conv2F_1 = Conv2DAdaptive(filters=nfilters,
                                kernel_size=(afw,afw),
                                data_format='channels_last',strides=1,
                                padding='same',name='acnn-1', activation='relu',
                                trainSigmas=True, trainWeights=True,
                                init_sigma=[0.1,0.5],
                                gain = 1.0,
                                kernel_regularizer=None,
                                sigma_regularizer=regularizers.l2(1e-5),
                                bias_initializer=initializers.Constant(0),
                                norm=2,use_bias=False)
        
        Conv2F_2 = Conv2DAdaptive(filters=nfilters,
                                kernel_size=(afw,afw),
                                data_format='channels_last',strides=1,
                                padding='same',name='acnn-2', activation='relu',
                                trainSigmas=True, trainWeights=True,
                                init_sigma=[0.1,0.5],
                                gain = 1.0,
                                kernel_regularizer=None,
                                bias_initializer=initializers.Constant(0),
                                norm=2,use_bias=False)
    elif settings['test_layer']=='conv':
        Conv2F_1 = Conv2D(filters=nfilters,
                         kernel_size=(afw,afw), padding='same',
                        activation='relu')
        Conv2F_2 = Conv2D(filters=nfilters,
                         kernel_size=(afw,afw), padding='same',
                        activation='relu')
    
    else:
        print("UNKNOWN TEST LAYER")
        return
    print(network[-1])
    network.append(Conv2F_1(network[-1]))
    # CONV2Adaptive works fine without batchnorm also. 
    #network.append(BatchNormalization()(network[-1]))
    #network.append(Activation('relu')(network[-1]))

    network.append(Conv2F_2(network[-1]))
    #network.append(BatchNormalization()(network[-1]))
    #network.append(Activation('relu')(network[-1]))


    network.append(MaxPooling2D((2,2))(network[-1]))
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

    model = Model(inputs=[network[0]], outputs=network[-1])

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


        conv_node = Conv2DAdaptive(filters=nfilters,kernel_size=(afw,afw),
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
    #network.append(ReLU(negative_slope=0.01)(network[-1]))t_test__patterns(nm_list,scr_list,[('aconv_9x9','conv_9x9')])

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

    model = tf.keras.models.Model(inputs=[network[0]], outputs=network[-1])

    return model



def test_mnist(settings,sid=9):
    

    #sess = K.get_session()
    K.clear_session()
    #K.set_session(sess)
    np.random.seed(sid)
    tf.random.set_seed(sid)


    #dset='cifar10'

    dset = settings['dset']
    batch_size = settings['batch']
    num_classes = 10
    epochs =settings['epochs']
    test_acnn = settings['test_layer']=='aconv'
    kernel_size = settings['adaptive_kernel_size']

    #ld_data = load_dataset(dset,normalize_data=True,options=[])
    #x_train,y_train,x_test,y_test,input_shape,num_classes=ld_data
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
        num_classes = target_names.shape[0]
        
        from sklearn.model_selection import train_test_split
        
        #X -= X.mean()
        #X /= X.std()
        #split into a training and testing set
        x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25)
        
        print("Total dataset size:")
        print("n_samples: %d" % n_samples)
        print("n_features: %d" % n_features)
        print("n_classes: %d" % num_classes)
               

        

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
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    if settings['arch']=='simple':
        model = create_2_layer_network(input_shape, num_classes, settings)
    else:
        model = create_4_layer_network(input_shape, num_classes, settings)
        
    model.summary()


 
    lr = 0.1 *settings['lr_multiplier']
    from tensorflow.keras.optimizers import Adam
    opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, clipvalue=1.0)
    #opt = tf.keras.optimizers.Adam(lr=1e-3, clipvalue=1.0)
                   
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    
  
    stat_func_name = ['max: ', 'mean: ', 'min: ', 'var: ', 'std: ']
    stat_func_list = [np.max, np.mean, np.min, np.var, np.std]
    #callbacks = [tb]
    callbacks = []
    
    
    red_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=10, min_lr=1e-5)
    
    callbacks+=[red_lr]
    MIN_SIG = 1.0/kernel_size
    MAX_SIG = kernel_size*1.0
    MIN_MU = 0.0
    MAX_MU = 1.0
    if  settings['test_layer']=='aconv':
        ccp1 = ClipCallback('Sigma',[MIN_SIG,MAX_SIG])
      
        
        #ccp = ClipCallback('Sigma',[MIN_SIG,MAX_SIG])
        pr_1 = PrintLayerVariableStats("acnn-1","Weights:0",stat_func_list,stat_func_name)
        pr_2 = PrintLayerVariableStats("acnn-1","Sigma:0",stat_func_list,stat_func_name)
        
        callbacks += [ccp1]
    
    #print("TRAINABLE WEIGHTS:",model.trainable_weights)


    tf.keras.utils.plot_model(
    model,
    to_file="model_simple.png",
    show_shapes=True,
    show_layer_names=True,
    expand_nested=True,
    dpi=300
    )

    #print(opt)
    #opt = SGD(lr=0.01,momentum=0.9)
    from tensorflow.keras.losses import categorical_crossentropy
    model.compile(loss=categorical_crossentropy,
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
            rotation_range=0,
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
    mx_scores = [np.max(list_histories[i].history['val_accuracy']) for i in range(len(list_histories))]
    histories = [h.history for h in list_histories]
    print("Max sscores", mx_scores)
    print("Max stats", np.mean(mx_scores), np.std(mx_scores))
    return mx_scores, histories, list_scores




if __name__ == '__main__':
    #import test_filters
    #exit()
    import sys
    kwargs = {'dset':'mnist', 'arch':'simple', 'repeats':1, 
              'test_layer':'conv',
              'epochs':20, 'batch':128, 'exname':'noname', 
              'adaptive_kernel_size':7, 'nfilters':32, 
              'data_augmentation':False, 'lr_multiplier':1.0}
    # For MNIST YOU CAN USE lr_multiplier 1.0, 
    # FOR OTHER USE lr_multiplier 0.1 or lower
    
    # default parameters runs MNIST. reaches ~99.65 val_accuracy @ 100 epochs.
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
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    filename = 'outputs/'+kwargs['exname']+'_'+kwargs['arch']+'_'+kwargs['dset']+'_'+kwargs['test_layer']+'_'+ks+'_'+timestr+'_'+'._results.npz'
    np.savez_compressed(filename,mod=kwargs,mx_scores =r[0], results=r)


