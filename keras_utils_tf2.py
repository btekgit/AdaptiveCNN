#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:02:20 2018

@author: btek
"""

from __future__ import print_function

import h5py
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

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

def get_layer_weights(model,layer_name,verbose=0):
    
    for layer in model.layers:
        if (layer.name==layer_name):
            g=layer.get_config()
            w = layer.get_weights()
            if verbose>0:
                print("Layer: ", layer)
                print('name:',layer.name)
                print(g)
                print(len(w))
            return w
        else:
            return None


def kdict(d,name_prefix=''):
    r = {}
    for k,v in d.items():
        #print("KEY VALUE PAIRS ",k,v)
        r[k] = K.variable(v, name=name_prefix+str(k).replace(':','_'))
    return r     
        
def eval_Kdict(d):
    '''evaluates all variables in a dictionary'''
    l = [str(k)+':'+str(v.numpy()) for k,v in d.items()]
    return l


def set_pattern_find(name, keyset):
    ''' this function searchs keyset patterns in name. 
    if finds it, it returns that key. single match only'''
    #print(name, keyset)
    #print(name, keyset)
    for k in keyset:
        if name.find(k) >= 0:
            print(name, keyset)
            return k   
    return None

def standarize_image_025(trn, val=None, tst=None):

    K = 4.0 # 2.0 is very good with MNIST 99.20-99.19
    M = 256.0
    trn /= M
    trn *=K
    trn -= K/2.0
    if val is not None:
        val /= M
        val *= K
        val -= K/2
    if tst is not None:
        tst /= M
        tst *= K
        tst -= K/2
    
    return trn, val, tst

def standarize_image_01(trn, val=None, tst=None):

    import numpy as np
    M = 255.0
    trn /= M
    K = np.mean(trn, axis=0)
    trn -= K
    if val is not None:
        val /= M
        val -= K
    if tst is not None:
        tst /= M
        tst -= K
    
    return trn, val, tst


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




class WeightHistory(Callback):

    def __init__(self, model, layername, verbose=0):
        self.batchlist=[]
        self.epochlist=[]
        self.sess = None
        self.warn = True
        self.model = model
        self.layername = layername
        self.verbose = verbose
        if verbose>0:
            print("Weight history set for: ", self.model.get_layer(self.layername))
        super(WeightHistory, self).__init__()
    
    def set_model(self, model):
        self.model = model
        #print(self.model.summary())
    
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
        



class RecordWeights(Callback):
    def __init__(self,name,var):
        self.layername = name
        self.varname = var
        
    def setVariableName(self,name, var):
        self.layername = name
        self.varname = var
    def on_train_begin(self, logs={}):
        self.record = []
        all_params = self.model.get_layer(self.layername)._trainable_weights
        all_weights = self.model.get_layer(self.layername).get_weights()

        for i,p in enumerate(all_params):
            #print(p.name)
            if (p.name.find(self.varname)>=0):
            #print("recording", p.name)
                self.record.append(all_weights[i])

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

class RecordOutput(Callback):
    def __init__(self,model,names,stat_functions, stat_names):
        self.output_layers = names
        self.s_name = 's'
        self.model = model
        self.model.metrics_names += [self.s_name]
        self.model.metrics_tensors += [layer.output for layer in model.layers if layer.name in self.output_layers]
        self.stat_list = stat_functions
        self.stat_names = stat_names
      
    def on_batch_begin(self, batch, logs=None):
        self.on_batch_end(batch, logs)
    def on_batch_end(self, batch, logs=None):
        
        print('log keys:', logs.keys())
        
        s_pred = logs[self.s_name]
        print('s_pred:', s_pred.shape)
        for i,p in enumerate(s_pred):
            print(i,p)
            stat_str = [n+str(s(p)) for s,n in zip(self.stat_list,self.stat_names)]
            print("Stats for", p.name, stat_str)

        return

class RecordVariable(RecordWeights):
        #print("The name for Record Variable has changed, use RecordWeights or RecordTensor instead")
        pass

class RecordTensor(Callback):
    pass
    def __init__(self,tensor, on_batch=True,  on_epoch=False):
        print("Not working!")
        self.tensor = tensor
        self.on_batch = on_batch
        self.on_epoch = on_epoch
    def setVariableName(self,tensor):
        self.layername = tensor
    def on_train_begin(self, logs={}):
        self.record = []
    def on_batch_end(self, batch, logs={}):        
        self.record.append(self.tensor.numpy())   
    def on_epoch_end(self,epoch, logs={}):
        self.record.append(self.tensor.numpy())


class PrintLayerVariableStats(Callback):
    def __init__(self,name,var,stat_functions,stat_names,not_trainable=False):
        
        self.layername = name
        self.varname = var
        self.stat_list = stat_functions
        self.stat_names = stat_names
        self.not_trainable = not_trainable

    def setVariableName(self,name, var):
        self.layername = name
        self.varname = var
    def on_train_begin(self, logs={}):
        all_params = self.model.get_layer(self.layername).weights
        all_weights = self.model.get_layer(self.layername).get_weights()

        for i,p in enumerate(all_params):
            #print(p.name)
            if (p.name.find(self.varname)>=0):
                stat_str = [n+str(s(all_weights[i])) for s,n in zip(self.stat_list,self.stat_names)]
                print("Stats for", p.name, stat_str)

        #def on_batch_end(self, batch, logs={}):
        #    self.record.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        all_params = self.model.get_layer(self.layername).weights
        
        all_weights = self.model.get_layer(self.layername).get_weights()
            
        for i,p in enumerate(all_params):
            #print(p.name)
            if (p.name.find(self.varname)>=0):
                stat_str = [n+str(s(all_weights[i])) for s,n in zip(self.stat_list,self.stat_names)]
                print("Stats for", p.name, stat_str)


class RecordFunctionOutput(Callback):
    def __init__(self,funct, avg=False):
        print("Not working!")
        self.funct = funct
        self.sess = K.get_session()
        self.avg=avg
        self.count = 0


    def setVariableName(self,funct):
        self.funct = funct
    def on_train_begin(self, logs={}):
        self.record = []
        self.count = 0
        if K.backend() == 'tensorflow':
            self.sess = K.get_session()
        
    def on_train_end(self, logs={}):
        if self.avg:
            self.record[0]/=self.count


    def on_batch_end(self, batch, logs={}):

        inp = logs['ins_batch'][0]
        acc = self.funct([inp])
        if self.avg and self.record:
                self.record[0]+=acc[0]




class PrintAnyVariable(Callback):

    def __init__(self,scope, varname):
        print("THIS DOES NOT WORK!!!!",)
        self.scope = scope
        self.varname = varname
            
    def setVariableName(self,scope, varname):
        self.scope = scope
        self.varname = varname
    def on_train_begin(self, logs={}):
        vs =[n.name for n in K.tf.get_default_graph().as_graph_def().node]
        print(vs)
        for v in vs:
            if v.name ==self.varname:
                print(":", K.get_value(v))

    def on_epoch_end(self, epoch, logs={}):
        g = K.tf.get_default_graph()
        v = g.get_tensor_by_name(self.varname)
        print(v)
        vs =[n for n in K.tf.get_default_graph().as_graph_def().node]
        for v in vs:
            if v.name ==self.varname:
                print(v.name)



from tensorflow.keras.optimizers import Optimizer
from six.moves import zip
#from keras.legacy import interfaces
    
class SGDwithLR(Optimizer):
    """Stochastic gradient descent optimizer with different LEARNING RATES
    CODED BY BTEK
    Uses dictionaries for different parameters.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    pattern_search_true allows dict keys to be patterns instead of exact match.

    # Arguments
        LR: a dictionary of floats for, float >= 0
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
        
    """
    def __init__(self, lr={'all':0.1}, momentum={'all':0.0}, decay={},
                 clips={}, decay_epochs=None,
                 nesterov=False, verbose=0, update_clip=100.0, 
                 pattern_search=False,**kwargs):
        super(SGDwithLR, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            if 'all' not in lr.keys():
                print('adding LR for all elements')
                lr.setdefault('all',0.1)
            if isinstance(lr, (float,int)):  
                print('This SGD works with dictionaries')
                lr = {'all',lr}
                #print(lr)
            
            if  isinstance(momentum, (float,int)):  
                print('This SGD works with dictionaries')
                momentum = {'all',momentum}
                #print(lr)
        
            self.lr = kdict(lr,'lr_')
            #print("LEARNING RATE: ", lr)
            self.momentum = kdict(momentum,'mom_')
            self.decay = kdict(decay,'dec_')
            self.clips = kdict(clips,'clips')
            self.clips_val = clips
            self.pattern_search = pattern_search
            if decay_epochs is not None:
                self.decay_epochs=K.variable(decay_epochs, dtype='int64')
            else:
                self.decay_epochs=[]
            
            #abs(new_p-old_p)/old_p < gc 
            self.UPCLIP =update_clip ## update can not be larger than this. 
            
                
                    
        self.nesterov = nesterov
        self.verbose = verbose

    
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)

        # first update the number of iterations
        self.updates = [K.update_add(self.iterations, 1)]
        
        if self.decay_epochs:
            ite_casted = K.cast(self.iterations, K.dtype(self.decay_epochs))
            hit_decay_epoch = K.any(K.equal(ite_casted, self.decay_epochs))
            
            #print(hit_decay_epoch)
            lr = K.switch(hit_decay_epoch, self.lr['all']*self.decay['all'],
                          self.lr['all'])

            #K.print_tensor(self.lr['all'])
            #a = K.switch(hit_decay_epoch, 
            #             K.print_tensor(self.lr['all'],message='Decays:'), 
            #             K.print_tensor(self.lr['all'],message=' '))


            self.updates.append(K.update(self.lr['all'],lr))

        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(s) for s in shapes]
        self.weights = [self.iterations] + moments
        #print(self.weights)
        

        for p, g, m in zip(params, grads, moments):
            #print("HEREEEE:", p.name, g, m)
            lrptrkey= set_pattern_find(p.name,self.lr.keys())
            if lrptrkey:
                if self.verbose>0:
                    print("Setting different learning rate for ", p.name, " : ", K.eval(self.lr[lrptrkey]))
                lr = self.lr[lrptrkey]
                dcptrkey=set_pattern_find(p.name,self.decay.keys())
                if self.decay_epochs and dcptrkey:
                    lr = K.switch(hit_decay_epoch, self.lr[lrptrkey]*self.decay[dcptrkey],
                                  self.lr[lrptrkey])
                    self.updates.append(K.update(self.lr[lrptrkey],lr))
                    if self.verbose>0:
                        print("Added decay to ", p.name, ": ", K.eval(lr),",",self.decay[dcptrkey])
                elif self.decay_epochs:
                    lr = K.switch(hit_decay_epoch, self.lr[lrptrkey]*self.decay['all'],self.lr[lrptrkey])
                    self.updates.append(K.update(self.lr[lrptrkey],lr))
                    if self.verbose>0:
                        print("Added decay to ", p.name, ": ", K.eval(lr),",",self.decay['all'])
                else:
                    lr = self.lr[lrptrkey]

            else:
                lr = self.lr['all']
            
            momptrkey = set_pattern_find(p.name,self.momentum.keys())
            if momptrkey:
                if self.verbose>0:
                    print("Setting different momentum for ", p.name, " , ", 
                          K.eval(self.momentum[momptrkey]))
                momentum = self.momentum[momptrkey]
            else:
                momentum = self.momentum['all'] 

            v = momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + momentum * (momentum * m - lr * g) - lr * g
            else:
                new_p = p + momentum * m - lr * g
            
            # CHANGE CLIP
            _to_tensor = K.tensorflow_backend._to_tensor
            _clip_by_val = K.tf.clip_by_value
            margin = K.mean(K.abs(p*K.constant(self.UPCLIP)))
            min_value = _to_tensor(p-margin, p.dtype.base_dtype)
            max_value = _to_tensor(p+margin, p.dtype.base_dtype)
            
            max_v = K.maximum(min_value, max_value)
            min_v = K.minimum(min_value, max_value)

            new_p = _clip_by_val(new_p, min_v, max_v)
             
            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            clptrkey = set_pattern_find(p.name,self.clips.keys())
            if self.clips_val and clptrkey:
                if self.verbose>0:
                    print("Clipping variable",p.name," to ", self.clips[clptrkey])
                c = K.eval(self.clips[clptrkey])
                new_p = K.clip(new_p, c[0], c[1])
            #print("updates for ", p.name, " lr: ", K.eval(lr), " mom:", K.eval(momentum))
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': str(eval_Kdict(self.lr)),
                  'momentum': str(eval_Kdict(self.momentum)),
                  'decay': str(eval_Kdict(self.decay)),
                  'clips': str(eval_Kdict(self.clips)),
                  'nesterov': self.nesterov}
        base_config = super(SGDwithLR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

class SGDwithCyclicLR(Optimizer):
    """Stochastic gradient descent optimizer with different LEARNING RATES and 
    CYCLING LEARNING RATE CHANGE. IT USES A GAUSSIAN PEAKING TO lr AT peaklriter iteratirons.
    CODED BY BTEK
    Uses dictionaries for different parameters.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    pattern_search_true allows dict keys to be patterns instead of exact match.
    "Cycling LR SGD OPTIMIZER: the learning starts from min_lr,",
                  " raises to lr in peaklriter epochs than drops down to min_lr ",
                  " in  peallriter iterations. Note that iterations are not epochs",
                  " but batch iterations"

    # Arguments
        LR: a dictionary of floats for, float >= 0
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
        
    """
    def __init__(self, peaklriter,lr={'all':0.01}, momentum={'all':0.0}, 
                 min_lr={'all':0.0001}, peak_lr={'all':2.0}, lrsigma=0.5,
                 clips={}, nesterov=False, verbose=0, update_clip=100.0, 
                 pattern_search=True,**kwargs):
        super(SGDwithCyclicLR, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            print("Cycling LR SGD OPTIMIZER: the learning starts from min_lr,",
                  " raises to lr in peaklriter epochs than drops down to min_lr ",
                  " in  peallriter iterations. Note that iterations are not epochs",
                  " but batch iterations")
            if 'all' not in lr.keys():
                print('adding LR for all elements')
                lr.setdefault('all',2.0)
            if isinstance(lr, (float,int)):  
                print('This SGD works with dictionaries')
                lr = {'all',lr}
                #print(lr)
            
            if  isinstance(momentum, (float,int)):  
                print('This SGD works with dictionaries')
                momentum = {'all',momentum}
                #print(lr)
        
            self.lr = kdict(lr,'lr_')
            self.min_lr = kdict(min_lr,'min_lr')
            self.peak_lr = kdict(peak_lr,'peak_lr')
            
            self.peaklriter = K.constant(peaklriter)
            self.lrsigma = K.constant(lrsigma)
            #print("LEARNING RATE: ", lr)
            self.momentum = kdict(momentum,'mom_')
            self.clips = kdict(clips,'clips')
            self.clips_val = clips
            self.pattern_search = pattern_search
            #abs(new_p-old_p)/old_p < gc 
            self.UPCLIP =update_clip ## update can not be larger than this. 
            
                
                    
        self.nesterov = nesterov
        self.verbose = verbose

    
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)

        # first update the number of iterations
        self.updates = [K.update_add(self.iterations, 1)]
        
        # Cycling Gaussian LR
        # I implement this lr_f = lambda x,b,c,s: b+ s*np.exp(-(x-c)**2/(c*0.5)**2)
        def gauss_lr(min_lr, max_lr, center, lrsigma,i):
            
            return (min_lr+ max_lr*K.exp(-(i-center)**2/(center*lrsigma)**2))
        
        ite_casted = K.cast(self.iterations, K.dtype(self.peaklriter))
        all_lr = gauss_lr(self.min_lr['all'], self.peak_lr['all'],
                              self.peaklriter,self.lrsigma,ite_casted)
        #current_lr = self.min_lr['all'] + 
        #self.peak_lr['all']*K.exp(((ite_casted-self.peaklriter)**2)/(self.dropsigma*self.peaklriter)**2)
        ############################################################################
        self.updates.append(K.update(self.lr['all'],all_lr))

        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(s) for s in shapes]
        self.weights = [self.iterations] + moments
        #print(self.weights)
        

        for p, g, m in zip(params, grads, moments):
            #print("HEREEEE:", p.name, g, m)
            lrptrkey= set_pattern_find(p.name,self.lr.keys())
            if lrptrkey:
                if self.verbose>0:
                    print("Setting different learning rate for ", p.name, " : ", K.eval(self.lr[lrptrkey]))
                if set_pattern_find(p.name,self.min_lr.keys()) and set_pattern_find(p.name,self.peak_lr.keys()):
                    p_lr = gauss_lr(self.min_lr[lrptrkey], self.peak_lr[lrptrkey],
                                          self.peaklriter,self.lrsigma,ite_casted)
                else:
                    p_lr = gauss_lr(self.min_lr['all'], self.peak_lr['all'],
                                          self.peaklriter,self.lrsigma,ite_casted)
            else:
                p_lr = self.lr['all']
                
            momptrkey = set_pattern_find(p.name,self.momentum.keys())
            if momptrkey:
                if self.verbose>0:
                    print("Setting different momentum for ", p.name, " , ", 
                          K.eval(self.momentum[momptrkey]))
                momentum = self.momentum[momptrkey]
            else:
                momentum = self.momentum['all'] 

            
            

            if self.nesterov:
                updt = momentum * (momentum * m - p_lr * g) - p_lr * g
            else:
                updt = momentum * m - p_lr * g
            
            # CHANGE CLIP
            _to_tensor = K.tensorflow_backend._to_tensor
            _clip_by_val = K.tf.clip_by_value
            margin = K.mean(K.abs(p))*K.constant(self.UPCLIP)
            #margin = K.mean(K.abs(p*K.constant(self.UPCLIP)))
            #min_value = _to_tensor(-margin, p.dtype.base_dtype)
            #max_value = _to_tensor(margin, p.dtype.base_dtype)
            
            #max_v = K.maximum(min_value, max_value)
            min_v = K.zeros_like(margin)
            updt_sign = K.sign(updt)
            updt_val = _clip_by_val(K.abs(updt), min_v, margin)
            
            v = updt_sign * updt_val  # velocity
            new_p = p + v
            
            
            self.updates.append(K.update(m, v))
            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            clptrkey = set_pattern_find(p.name,self.clips.keys())
            if self.clips_val and clptrkey:
                c = K.eval(self.clips[clptrkey])
                if self.verbose>0:
                    print("Clipping variable",p.name," to ", c)
                    #input()
                new_p = K.clip(new_p, c[0], c[1])
            #print("updates for ", p.name, " lr: ", K.eval(lr), " mom:", K.eval(momentum))
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'peaklriter':self.peaklriter,'dropsigma': self.dropsigma,
                  'lr':self.lr, 'momentum':self.momenturm, 
                 'min_lr':self.min_Lr, 'peak_lr':self.peak_lr,
                 'clips':self.clips, 'nesterov': self.nesterov, 
                 'update_clip':self.UPCLIP}
        base_config = super(SGDwithLR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

class RMSpropwithClip(Optimizer):
    """RMSProp optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).

    This optimizer is usually a good choice for recurrent
    neural networks.

    # Arguments
        lr: float >= 0. Learning rate.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [rmsprop: Divide the gradient by a running average of its recent magnitude]
          (http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    """

    def __init__(self, lr=0.001, rho=0.9, epsilon=None, decay=0., clips={},
                 verbose=0,
                 **kwargs):
        super(RMSpropwithClip, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.rho = K.variable(rho, name='rho')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.clips = kdict(clips,'clips')
            self.clips_val = clips
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.verbose=verbose

    
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        accumulators = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        for p, g, a in zip(params, grads, accumulators):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))
            new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            
            clptrkey = set_pattern_find(p.name,self.clips.keys())
            if self.clips_val and clptrkey:
                if self.verbose>0:
                    print("CLpping variable",p.name," to ", self.clips[p.name] )
                c = K.eval(self.clips[clptrkey])
                new_p = K.clip(new_p, c[0], c[1])

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'rho': float(K.get_value(self.rho)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(RMSpropwithClip, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AdamwithClip(Optimizer):
    """Adam optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".

    # References
        - [Adam - A Method for Stochastic Optimization]
          (https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond]
          (https://openreview.net/forum?id=ryQu7f-RZ)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, clips={},
                 verbose=0, **kwargs):
        super(AdamwithClip, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.clips = kdict(clips,'clips')
            self.clips_val = clips
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.verbose = verbose

    
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
                
            clptrkey = set_pattern_find(p.name,self.clips.keys())
            if self.clips_val and clptrkey:            
                c = K.eval(self.clips[clptrkey])
                if self.verbose>0:
                    print("Clipping variable",p.name," to ", c )
                new_p = K.clip(new_p, c[0], c[1])

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad,
                  'clips': self.clips_val}
        base_config = super(AdamwithClip, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))