#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:02:20 2018

@author: btek
"""

from __future__ import print_function

import h5py
from keras.callbacks import Callback
import keras.backend as K

def print_layer_names(model):
    for layer in model.layers:
        print(layer)
        print(layer.name)
        print(layer.name=='gauss')
        
def print_layer_weights(model):
    for layer in model.layers:
        print(layer)
        print(layer.name)
        g=layer.get_config()
        print(g)
        w = layer.get_weights()
        print(len(w))
        print(w)

def get_layer_weights(model,layer_name):
    
    for layer in model.layers:
        if (layer.name==layer_name):
            print("Layer: ", layer)
            print('name:',layer.name)
            g=layer.get_config()
            print(g)
            w = layer.get_weights()
            print(len(w))
            return w
        else:
            return None
        

class WeightHistory(Callback):

    def __init__(self, model, layername):
        self.batchlist=[]
        self.epochlist=[]
        self.sess = None
        self.warn = True
        self.model = model
        self.layername = layername
        
        print("Weight history set for: ", self.model.get_layer(self.layername))
        super(WeightHistory, self).__init__()
    
    def set_model(self, model):
        self.model = model
        print(self.model.summary())
    
    def on_train_begin(self, logs={}):
        self.batchlist = []
        self.epochlist = []
        if K.backend() == 'tensorflow':
            self.sess = K.get_session()

#    def on_batch_end(self, batch, logs={}):
##        gauss_layer = self.model.get_layer(self.layername)  
##        gauss_layer_var = gauss_layer.get_weights()
##        #warn = True
##        if len(self.batchlist)< 10000:
##            self.batchlist.append(gauss_layer_var[0])
    
    def on_epoch_begin(self, batch, logs={}):
        gauss_layer = self.model.get_layer(self.layername)
        gauss_layer_var = gauss_layer.get_weights()
        #print("yes called")
        #warn = True
        if len(self.epochlist)< 10000:
            self.epochlist.append(gauss_layer_var[0])
                
        
    def get_batchlist(self):
        return self.batchlist
    
    def get_epochlist(self):
        return self.epochlist
        

def dump_keras_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                subkeys = param.keys()
                for k_name in param.keys():
                    print("      {}/{}: {}".format(p_name, k_name, param.get(k_name)[:]))
    finally:
        f.close()
        

class RecordVariable(Callback):
        def __init__(self,name,var):
            self.layername = name
            self.varname = var
        
        def setVariableName(self,name, var):
            self.layername = name
            self.varname = var
        def on_train_begin(self, logs={}):
            self.record = []

        #def on_batch_end(self, batch, logs={}):
        #    self.record.append(logs.get('loss'))
            
        def on_epoch_end(self,epoch, logs={}):
            all_params = self.model.get_layer(self.layername)._trainable_weights
            all_weights = self.model.get_layer(self.layername).get_weights()
            
            for i,p in enumerate(all_params):
                #print(p.name)
                if (p.name.find(self.varname)>=0):
                    #print("recording", p.name)
                    self.record.append(all_weights[i])
                    
                    
class PrintVariableStats(Callback):
        def __init__(self,name,var,stat_functions,stat_names):
            self.layername = name
            self.varname = var
            self.stat_list = stat_functions
            self.stat_names = stat_names
        
        def setVariableName(self,name, var):
            self.layername = name
            self.varname = var
        def on_train_begin(self, logs={}):
            all_params = self.model.get_layer(self.layername)._trainable_weights
            all_weights = self.model.get_layer(self.layername).get_weights()
            
            for i,p in enumerate(all_params):
                #print(p.name)
                if (p.name.find(self.varname)>=0):
                    stat_str = [n+str(s(all_weights[i])) for s,n in zip(self.stat_list,self.stat_names)]
                    print("Stats for", p.name, stat_str)

        #def on_batch_end(self, batch, logs={}):
        #    self.record.append(logs.get('loss'))
            
        def on_epoch_end(self, epoch, logs={}):
            all_params = all_weights = self.model.get_layer(self.layername)._trainable_weights
            all_weights = self.model.get_layer(self.layername).get_weights()
            
            for i,p in enumerate(all_params):
                #print(p.name)
                if (p.name.find(self.varname)>=0):
                    stat_str = [n+str(s(all_weights[i])) for s,n in zip(self.stat_list,self.stat_names)]
                    print("Stats for", p.name, stat_str)
                    

from keras.optimizers import Optimizer

from six.moves import zip


from keras.utils.generic_utils  import serialize_keras_object
from keras.utils.generic_utils import deserialize_keras_object
from keras.legacy import interfaces
    
class SGDwithLR(Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, **kwargs):
        super(SGDwithLR, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(SGDwithLR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))