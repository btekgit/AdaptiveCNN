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
from keras import backend as K
from keras.engine.topology import Layer
from keras.utils import conv_utils
from keras import activations, regularizers, constraints
from keras import initializers
from keras.engine import InputSpec
import numpy as np
import tensorflow as tf

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

@tf.custom_gradient
def samplefocus(x):
  y = K.random_binomial(shape=x.shape,p=x, dtype=x.dtype)
  def grad(dy):
    return dy 
  return y, grad


@tf.custom_gradient
def samplefocus2(x):
  y = tf.round(x)
  def grad(dy):
    return dy 
  return y, grad  

class Conv2DAdaptive(Layer):
    def __init__(self, rank, nfilters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_regularizer=None,
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
        super(Conv2DAdaptive, self).__init__(**kwargs)
        #def __init__(self, num_filters, kernel_sigmaze, incoming_channels=1, **kwargs):
        self.rank = rank
        self.nfilters = nfilters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = data_format
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
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
        self.Sigma =None
        self.norm = norm
                 
        #self.input_shape = input_shape
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'))
        print(kwargs)
        self.kernel_size = kernel_size
        
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
        
        self.Sigma = self.add_weight(shape=(self.nfilters,),
                                          name='Sigma',
                                          initializer=self.sigma_initializer,
                                          trainable=self.trainSigmas,
                                          constraint= constraints.NonNeg(),
                                          regularizer=None)
        
        self.W = self.add_weight(shape=[kernel_size[0],kernel_size[1],
                                        self.input_channels,self.nfilters],
                                 name='Weights',
                                 initializer=self.weight_initializer,
                                 #initializer=initializers.he_uniform(),
                                 trainable=True,
                                 regularizer = self.kernel_regularizer,
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
        
        super(Conv2DAdaptive, self).build(input_shape)  # Be sure to call this somewhere!
        

    
    def U(self):
  
        up= K.sum((self.idxs - self.mu)**2, axis=1)
        #print("up.shape",up.shape)
        up = K.expand_dims(up,axis=1,)
        #print("up.shape",up.shape)
        # clipping scaler in range to prevent div by 0 or negative cov. 
        sigma = K.clip(self.Sigma,0.01,5.0)
        #cov_scaler = self.cov_scaler
        dwn = 2 * ( sigma ** 2)
        #scaler = (np.pi*self.cov_scaler**2) * (self.idxs.shape[0])
        result = K.exp(-up / dwn)
        
        #print("-----------------------------------------------")
        #print(K.eval(result))
        #print("-----------------------------------------------")
        # Transpose is super important.
        #filter: A 4-D `Tensor` with the same type as `value` and shape
        #`[height, width, in_channels,output_channels]`
        # we do not care about input channels

        masks = K.reshape(result,(self.kernel_size[0], 
                                  self.kernel_size[1],
                                  1,self.nfilters))
        
        print("-----------------------------------------------")
        #print(K.eval(masks[:,:,0,0]))
        print(K.eval(masks[:,:,0,1]))
        print("-----------------------------------------------")
        
        if self.input_channels>1:
            masks = K.repeat_elements(masks, self.input_channels, axis=2)
            #print("U masks reshaped :",masks.shape)
        #print("inputs shape",inputs.shape)

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
        #print("Calling self.U:")
        self.kernel=None
        u = self.U()
        kernel = samplefocus2(u)
        
        print(K.eval(u))
        print(K.eval(kernel))
        # multiply with weights
        fiw = kernel*self.W
        print(K.eval(fiw))
        
        
        #---------------------------------------------------------------------
        #print("Trainable weights", self._trainable_weights)
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

        print("Scale initializer:",sigma)
        return sigma.astype(dtype)

        
    def weight_initializer(self,shape, dtype='float32'):
        #only implements channel last and HE uniform
        initer = 'He'
        distribution = 'uniform'
        
        kernel = K.eval(self.U())
        print("Kernel max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
        W = np.zeros(shape=shape, dtype=dtype)
        print("kernel shape:", kernel.shape, ", W shape: ",W.shape)
        # for Each Gaussian initialize a new set of weights
        verbose=True
        fan_out = np.prod(self.nfilters)*self.kernel_size[0]*self.kernel_size[1]
        
        for c in range(W.shape[-1]):
            #fan_in = np.sum((kernel[:,:,:,c])**2)
            fan_in = np.sum(kernel[:,:,:,c])
            
            #fan_in *= self.input_channels no need for this in repeated U. 
            if initer == 'He':
                std = self.gain * sqrt32(2.0) / sqrt32(fan_in)
            else:
                std = self.gain * sqrt32(1.0) / sqrt32(fan_in+fan_out)
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
    
    
    


def test():
    import os
    os.environ['CUDA_VISIBLE_DEVICES']="0"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"
    from keras.losses import mse
    import keras
    from keras.datasets import mnist,fashion_mnist, cifar10
    from keras.models import Sequential, Model
    from keras.layers import Input, Dense, Dropout, Flatten, Conv2D
    from skimage import filters
    from keras import backend as K
    from keras_utils import WeightHistory as WeightHistory
    from keras_utils import RecordVariable, PrintLayerVariableStats, SGDwithLR
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
   # Create a session with the above options specified.
    K.tensorflow_backend.set_session(tf.Session(config=config))
    K.clear_session()


    sid = 9
    # restarting everything requires sess.close()
    #g = tf.Graph()
    #sess = tf.InteractiveSession(graph=g)
    #sess = tf.Session(graph=g)
    #K.set_session(sess)
    np.random.seed(sid)
    tf.random.set_random_seed(sid)
    tf.compat.v1.random.set_random_seed(sid)
    
    from datetime import datetime
    now = datetime.now()
    timestr = now.strftime("%Y%m%d-%H%M%S")
    logdir = "tf_logs_cluttered/.../" + timestr + "/"
    
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    inputimg = x_train[0]/255
    sh = (inputimg.shape[0],inputimg.shape[1],1)
    outputimages = np.zeros(shape=[inputimg.shape[0],inputimg.shape[1],3],dtype='float32')
    outputimages[:,:,0] = filters.gaussian(inputimg,sigma=1)
    outputimages[:,:,1] = filters.sobel_h(inputimg)
    outputimages[:,:,2] = filters.sobel_v(filters.gaussian(inputimg,sigma=0.5))
    
    y = y_train[0]
    
    node_in = Input(shape=sh, name='inputlayer')
    # smaller initsigma does not work well. 
    node_acnn = Conv2DAdaptive(rank=2,nfilters=3,kernel_size=(5,5), 
                             data_format='channels_last',
                             strides=1,
                             padding='same',name='acnn',activation='linear',
                             init_sigma=0.5, trainSigmas=True, 
                             trainWeights=True,norm=2)(node_in)
    
    #node_acnn = Conv2D(filters=3,kernel_size=(7,7), 
    #                         data_format='channels_last',
    #                         padding='same',name='acnn',activation='linear')(node_in)
    
    
    model = Model(inputs=node_in, outputs=[node_acnn])
        
   # model.summary()

    from lr_multiplier import LearningRateMultiplier
    from keras.optimizers import SGD, Adadelta
    lr_dict = {'all':0.1,'acnn/Sigma:0': 0.1,'acnn/Weights:0': 0.1,
               'acnn-2/Sigma:0': 0.0001,'acnn-2/Weights:0': 0.01}
    
    mom_dict = {'all':0.9,'acnn/Sigma:0': 0.9,'acnn/Weights:0': 0.9,
                'acnn-2/Sigma:0': 0.9,'acnn-2/Weights:0': 0.9}
    clip_dict = {'acnn/Sigma:0': [0.1, 2.0]}
    # WORKS WITH OWN INIT With glorot uniform
    #print("WHAT THE ", lr_dict)
    opt = SGDwithLR(lr=lr_dict, momentum = mom_dict, clips=clip_dict,clipvalue=0.01)
    #(lr_dict, mom_dict) 
    #
    model.compile(loss=mse, optimizer=opt, metrics=None)
    model.summary()
    
    
    
    inputimg2 = np.expand_dims(np.expand_dims(inputimg,axis=0), axis=3)
    outputimages2 = np.expand_dims(outputimages,axis=0)
    
    #print info about weights
    acnn_layer = model.get_layer('acnn')    
    all_params=acnn_layer.weights
    print("All params:",all_params)
    acnn_params = acnn_layer.get_weights()
    for i,v in enumerate(all_params):
        print(v, ", max, mean", np.max(acnn_params[i]),np.mean(acnn_params[i]),"\n")
    
    
    mchkpt = keras.callbacks.ModelCheckpoint('models/weights.txt', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    wh0= WeightHistory(model, "acnn")

    sigma_history = []
    sigma_call = lambda x, batch=1, logs={}: x.append(acnn_layer.get_weights()[0])
    cov_dump = keras.callbacks.LambdaCallback(on_epoch_end=sigma_call(sigma_history))
    
    
    
    rv = RecordVariable("acnn","acnn/Sigma:0")
        
    history = model.fit(inputimg2, outputimages2,
              batch_size=1,
              epochs=1000,
              verbose=1, callbacks=[wh0,cov_dump,rv])
    
   
    print ("RECORD", len(rv.record), rv.record[0].shape)
    
    import matplotlib.pyplot as plt 
    rv_arr = np.array(rv.record)
    plt.plot(rv_arr)
    plt.title("Sigma")
    
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title("Loss")

    
    #print info about weights
    print("After training:",all_params)
    acnn_params = acnn_layer.get_weights()
    for i,v in enumerate(all_params):
        print(v, ", max, mean", np.max(acnn_params[i]),np.mean(acnn_params[i]),"\n")
    
    # print info about sigmas
    #print("Recorded sigma history", sigma_history)
    #print("Recorded weight history", wh0.get_epochlist())
    
    
    pred_images = model.predict(inputimg2,  verbose=1)
    print("Prediction shape",pred_images.shape)
    plt = True
    if plt:
        print("Plotting kernels before...")
        import matplotlib.pyplot as plt
        num_images=min(pred_images.shape[3],12)
        fig=plt.figure(figsize=(10,5))
        plt.subplot(3, num_images, 2)
        plt.imshow(np.squeeze(inputimg2[0,:,:,0]))
        for i in range(num_images):
            plt.subplot(3, num_images, i+4)
            plt.imshow(np.squeeze(outputimages2[0,:,:,i]))
            plt.title("output image")
            print("Max-in:",i," ",np.max(np.squeeze(outputimages2[0,:,:,i])))
        
        for i in range(num_images):
            plt.subplot(3, num_images, i+7)
            plt.imshow(np.squeeze(pred_images[0,:,:,i]))
            plt.title("pred image")
            print("MAx:","pred",i,np.max(np.squeeze(pred_images[0,:,:,i])))
            
        plt.show()
        plt.figure()
        for i in range(num_images):
            plt.subplot(3, num_images, i+1)
            #print(acnn_params[1].shape)
            plt.imshow(np.squeeze(acnn_params[1][:,:,0,i]))
            #print("MAx:","pred",i,np.max(np.squeeze(acnn_params[i])))
        #fig.colorbar(im, ax=ax1)
        plt.show()
        
        
        plt.figure()
        for i in range(num_images):
            plt.subplot(3, num_images, i+1)
            print("U -shape: ", acnn_layer.U().shape,type(K.eval(acnn_layer.U()[:,:,0,i])))
            print("Prod-shape", (acnn_params[1][:,:,0,i]*acnn_layer.U()[:,:,0,i]).shape)
            plt.imshow(np.float32(K.eval(acnn_layer.U()[:,:,0,i])))
            plt.title("U func")
        plt.figure()
        for i in range(num_images):
            plt.subplot(3, num_images, i+1)
            print("U -shape: ", acnn_layer.U().shape,type(K.eval(acnn_layer.U()[:,:,0,i])))
            print("Prod-shape", (acnn_params[1][:,:,0,i]*acnn_layer.U()[:,:,0,i]).shape)
            plt.imshow(np.float32(acnn_params[1][:,:,0,i]*K.eval(acnn_layer.U()[:,:,0,i])))
            plt.title("Weights")

            #plt.imshow()
            
            #print("MAx:","pred",i,np.max(np.squeeze(acnn_params[i])))
        #fig.colorbar(im, ax=ax1)
        plt.show()
        
        
        
    #print( model.get_layer('acnn').output )
    print( "Final Sigmas", model.get_layer('acnn').get_weights()[0] )
    
    K.clear_session()

#tf.reset_default_graph()
#test()

#test()
    
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

def test_mnist():     
    import os
    os.environ['CUDA_VISIBLE_DEVICES']="0"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    import keras
    from keras.datasets import mnist,cifar10
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
    from keras_utils import RecordVariable, PrintLayerVariableStats, SGDwithLR
    
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.1
   # Create a session with the above options specified.
    #K.tensorflow_backend.set_session(tf.Session(config=config))
    
    sid = 9
    #sess = K.get_session()
    K.clear_session()
    #sess = tf.Session(graph=g)
    #K.set_session(sess)
    np.random.seed(sid)
    tf.random.set_random_seed(sid)
    tf.compat.v1.random.set_random_seed(sid)
    
    
    #dset='cifar10'
    
    dset = 'mnist'
    batch_size = 512
    num_classes = 10
    epochs =200
    test_acnn = True
    regulazer = None
    prfw = 5
    fw = 5
    residual = False
    data_augmentation = False
    
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
    
    network=[]
    network.append(Input(shape=input_shape))

#    if test_acnn:
#        
#        prev_layer = network[-1]
#        
#        
#        conv_node = Conv2DAdaptive(rank=2,nfilters=32,kernel_size=(fw,fw), 
#                                data_format='channels_last',strides=1,
#                                padding='same',name='acnn-1', activation='linear',
#                                trainSigmas=True, trainWeights=True, 
#                                init_sigma=[0.15,1.0],
#                                gain = np.sqrt(1.0),
#                                kernel_regularizer=None,
#                                init_bias=initializers.Constant(0))(prev_layer)
#        if residual:
#            network.append(Add()([conv_node,prev_layer]))
#        else:
#            network.append(conv_node)
#        #, input_shape=input_shape))
#    else:
#        
#        network.append(Conv2D(24, (fw, fw), activation='linear', 
#                         kernel_initializer=w_ini(), 
#                         kernel_regularizer=None,
#                         padding='same')(network[-1]))
        
    network.append(Conv2D(32, kernel_size=(prfw, prfw),
                       activation='linear',padding='same', kernel_initializer=w_ini(),
                       kernel_regularizer=regulazer)(network[-1]))
    network.append(BatchNormalization()(network[-1]))
    network.append(Activation('relu')(network[-1]))
    network.append(Dropout(0.2)(network[-1]))
    network.append(Conv2D(32, (prfw, prfw), activation='linear', 
                     kernel_initializer=w_ini(), padding='same',
                     kernel_regularizer=regulazer)(network[-1]))
    #odel.add(MaxPooling2D(pool_size=(2, 2)))
    network.append(BatchNormalization()(network[-1]))
    network.append(Activation('relu')(network[-1]))
    
    network.append(Dropout(0.2)(network[-1]))
    
#    model.add(Conv2D(32, (3, 3), activation='linear', 
#                     kernel_initializer=w_ini(), padding='same',
#                     kernel_regularizer=regulazer))
#    #odel.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(BatchNormalization())
#    model.add(Activation('relu'))
#    
#    model.add(Dropout(0.2))
#    
    #odel.add(Dense(128, activation='relu'))
    
    #odel.add(Dropout(0.25))
    
      #=============================================================================
    nfilter= 32
    if test_acnn:
        
        prev_layer = network[-1]
        
        
        conv_node = Conv2DAdaptive(rank=2,nfilters=nfilter,kernel_size=(fw,fw), 
                                data_format='channels_last',strides=1,
                                padding='same',name='acnn-1', activation='linear',
                                trainSigmas=True, trainWeights=True, 
                                init_sigma=[0.25,0.75],
                                gain = 1.0,
                                kernel_regularizer=None,
                                init_bias=initializers.Constant(0),
                                norm=2)(prev_layer)
        if residual:
            network.append(Add()([conv_node,prev_layer]))
        else:
            network.append(conv_node)
        #, input_shape=input_shape))
    else:
        #fw = 7
        #v_ini = VS_ini(scale=0.25,mode='fan_in',distribution='uniform')
        network.append(Conv2D(nfilter, (fw, fw), activation='linear', 
                         kernel_initializer=w_ini(), 
                         kernel_regularizer=None,
                         padding='same')(network[-1]))
        #, input_shape=input_shape))
        
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
                   kernel_regularizer=regulazer)(network[-1]))
    network.append(BatchNormalization()(network[-1]))
    network.append(ReLU()(network[-1]))
    network.append(Dropout(0.2)(network[-1]))
    
    
    network.append(Dense(num_classes, activation='softmax',
                    kernel_regularizer=regulazer)(network[-1]))
    model = keras.models.Model(inputs=[network[0]], outputs=network[-1])
    model.summary()
    print("MAY BE MAXPOOL LAYER IS AFFECTING SIGNAL ")
    
    
    from lr_multiplier import LearningRateMultiplier
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
    lr_dict = {'all':0.1,'acnn/Sigma:0': 0.1,'acnn/Weights:0': 0.1,
               'acnn-2/Sigma:0': 0.0001,'acnn-2/Weights:0': 0.01}
    
    mom_dict = {'all':0.9,'acnn/Sigma:0': 0.9,'acnn/Weights:0': 0.9,
                'acnn-2/Sigma:0': 0.9,'acnn-2/Weights:0': 0.9}
    clip_dict = {'acnn/Sigma:0': [0.1, 2.0]}
    #print("WHAT THE ", lr_dict)
    opt = SGDwithLR(lr=lr_dict, momentum = mom_dict, clips=clip_dict,clipvalue=0.01)
        
    e_i = x_train.shape[0] // batch_size
        
    #decay_epochs =np.array([e_i*10], dtype='int64') #for 20 epochs
    #decay_epochs =np.array([e_i*10,e_i*80,e_i*120,e_i*160], dtype='int64')
    
    #opt = SGDwithLR(lr_dict, mom_dict,decay_dict,clip_dict, decay_epochs)#, decay=None)
    #opt= Adadelta()
    #lr_scheduler = LearningRateScheduler(lr_schedule,lr)
    
    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    if test_acnn:
        model_name = '%s_acnn%dx_model.{epoch:03d}.h5'% (dset, fw)
    else:
        model_name = '%s_cnn%dx_model.{epoch:03d}.h5'% (dset, fw)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    filepath = os.path.join(save_dir, model_name)
    print("Saving in ", filepath)

#    # Prepare callbacks for model saving and for learning rate adjustment.
#    checkpoint = ModelCheckpoint(filepath=filepath,
#                             monitor='val_acc',
#                             verbose=1,
#                             save_best_only=True)
    chkpt= keras.callbacks.ModelCheckpoint('best-model.h5', 
                                    monitor='val_acc', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    save_weights_only=True, 
                                    mode='max', period=1)
    
    
    tb = TensorBoard(log_dir='./tb_logs/mnist/acnn-res-lr5',
                     histogram_freq = 1, 
                     write_grads=True,
                     write_graph=False)
    
    stat_func_name = ['max: ', 'mean: ', 'min: ', 'var: ', 'std: ']
    stat_func_list = [np.max, np.mean, np.min, np.var, np.std]
    callbacks = [tb]
    callbacks = []
    
    if test_acnn:
        pr_1 = PrintLayerVariableStats("acnn-1","Weights:0",stat_func_list,stat_func_name)
        pr_2 = PrintLayerVariableStats("acnn-1","Sigma:0",stat_func_list,stat_func_name)
        rv_weights_1 = RecordVariable("acnn-1","Weights:0")
        rv_sigma_1 = RecordVariable("acnn-1","Sigma:0")
        callbacks+=[pr_1,pr_2,rv_weights_1,rv_sigma_1]
    else:
        pr_1 = PrintLayerVariableStats("conv2d_3","kernel:0",stat_func_list,stat_func_name)
        rv_weights_1 = RecordVariable("conv2d_3","kernel:0")
        callbacks+=[pr_1, rv_weights_1]
    pr_3 = PrintLayerVariableStats("conv2d_1","kernel:0",stat_func_list,stat_func_name)
    rv_kernel = RecordVariable("conv2d_1","kernel:0")
    callbacks+=[pr_3,rv_kernel]
    
    print("CALLBACKS:",callbacks)
    
    print("TRAINABLE WEIGHTS:",model.trainable_weights)
    
    print("WARNING by BTEK: if you see An operation has `None` for gradient. \
          Please make sure that all of your ops have a gradient defined \
          (i.e. are differentiable). Common ops without gradient: \
              K.argmax, K.round, K.eval. REMOVE TENSORBOARD CALLBACK OR EDIT IT!")
    
    
    #print(opt)
    #opt = SGD(lr=0.01,momentum=0.9)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    
    plt = False
    if plt and test_acnn:
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
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
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
            zoom_range=0.,
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
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=1, workers=4,
                            callbacks=callbacks, 
                            steps_per_epoch=x_train.shape[0]//batch_size)
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    
    if plt and test_acnn:
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
        
        
        
#tf.reset_default_graph()
#test()
test_mnist()




