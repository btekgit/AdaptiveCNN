#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:46:56 2020

@author: btek
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 17:04:12 2020

@author: btek
"""

import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
tfc = tf.constant
input_channels = 5
def func_emd(x):
    #return c-d*(np.exp(-((x-a)**2)/(d**2) ))+e*x**2+f*x**3
    #return c-d*math.pi*(np.exp(-((x-a)**2)/(b) ))
    k = np.array([5.69576177e-05, 5.70944988e-01, 9.29715380e-01, 2.26213386e+00])
    bas = k[3]/(math.sqrt(2*math.pi))
    return k[2]-bas*(np.exp(-((x-k[0])**2)/(k[1]**2) ))+0.1

def calcU(idxs, mu, si,kernel_size, nfilters):
    up= tf.reduce_sum((idxs - mu)**2, axis=1)
    print("up.shape",up.shape)
    up = tf.expand_dims(up,axis=1)
    sigma = tf.clip_by_value(si,1e-3,5.0)
    #sigma = si+1e-3
    '''
    a,b,c,d,e =  tfc(0.49699071),tfc(2.08261273), tfc(-2.51239034),tfc(0.79320175),tfc(-0.0631832352338566)
    s_pow2 = sigma * sigma
    s_pow3 = s_pow2 * sigma
    s_pow4 = s_pow3 * sigma
    '''
    #var_compens= func_emd(sigma)
    dwn = 2 * ( sigma ** 2)
    print("dwn.shape",dwn.shape)
    #scaler = (np.pi*self.cov_scaler**2) * (self.idxs.shape[0])
    #result = tf.exp(-up / dwn)/tf.sqrt(1-(tf.exp(-sigma))**tf.sqrt(2.0))
    #print(sigma, var_compens)
    #input()
    result = tf.exp(-up / dwn)#/tf.sqrt(var_compens)
    
    print("result shape",result.shape)
    masks = tf.reshape(result,(kernel_size,
                               kernel_size,
                               1,nfilters))
    
    masks = tf.repeat(masks, input_channels, axis=2)
    
    masks /= tf.sqrt(tf.reduce_sum(masks**2, axis=(0, 1, 2),keepdims=True))
    masks *= tf.sqrt(tf.constant(input_channels*kernel_size*kernel_size,dtype=tf.float32))

    return masks

def weight_initializer(kernel, shape, kernel_size, nfilters, gain=1.0,dtype='float32'):
     #only implements channel last and HE uniform
     initer = 'glorot'
     distribution = 'uniform'
     sqrt32 = math.sqrt
     print("Kernel max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
     W = np.zeros(shape=shape, dtype=dtype)
     print("kernel shape:", kernel.shape, ", W shape: ",W.shape)
     # for Each Gaussian initialize a new set of weights
     verbose=True
     fan_out = nfilters*kernel_size*kernel_size

     for c in range(W.shape[-1]):
         fan_in = np.sum((kernel[:,:,:,c])**2)

         #fan_in *= self.input_channels no need for this in repeated U.
         if initer == 'He':
             std = gain * math.sqrt(2.0) / math.sqrt(fan_in)
         else:
             std = gain * math.sqrt(2.0) / math.sqrt(fan_in+fan_out)
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


     return tf.convert_to_tensor(W)
 
input_channels = 25
nfilters =32
kernel_size = 7
wdist = 'winit'


mu = tf.convert_to_tensor(np.array([0.5, 0.5]), dtype=tf.float32, name='mu')

x,y= np.meshgrid(np.linspace(0.0,1.0,kernel_size),np.linspace(0.0,1.0,kernel_size))

idxs= tf.convert_to_tensor(np.array([x.flatten(), y.flatten()]).T, dtype=tf.float32,name='idx')


sis = np.linspace(0.00,2.5,nfilters)
plt.figure(231)
plt.figure(232,figsize=(32,5))
plt.figure(233,figsize=(32,5))
varlistUW=[]
varlistW=[]
k = 1
for s in sis:
    
    
    si = tf.random.uniform((1,nfilters),s*0.99,s*1.01,dtype='float32', name='si')
    #si =  tf.Variable(np.array([math.pow(2,-x+1) for x in range(neurons)],np.float32))
    #U=tf.exp((-(idx[:,tf.newaxis]-mu)**2)/(si**2)) / np.sqrt(1.0-np.exp(-s))
    U = calcU(idxs, mu, si, kernel_size, nfilters)
    #U = U / tf.reduce_sum(U,axis=0)
    #mn_U = tf.reduce_mean(U,axis=0)/U.shape[0]
    #print("Means", mn_U.numpy())
    if wdist=='normal':
        W = tf.random.normal(shape=[kernel_size,kernel_size,
                             input_channels,nfilters], 
                             mean=0, stddev=1.0,name='W')

    elif wdist=='uniform':
        W = tf.random.uniform([kernel_size,kernel_size,
                             input_channels,nfilters],-np.sqrt(3.0), np.sqrt(3.0),name='W')
    elif wdist=='delta':
        W = np.zeros([kernel_size,kernel_size,
                             input_channels,nfilters])
        for i in range(nfilters):
            print(i)
            W[np.int32(mu*kernel_size),
              np.int32(mu*kernel_size),input_channels//2,i] = kernel_size/2.0/np.max(U)# #np.sqrt(2.0/ISize) 
        W = tf.Variable(W, dtype=tf.float32)
    elif wdist=='winit':
        W = weight_initializer(U,[kernel_size,kernel_size,
                             input_channels,nfilters],kernel_size,nfilters,1.0)
        #W =  tf.Variable(tf.initializers.glorot_uniform()([kernel_size,kernel_size,input_channels,nfilters]))#dtype=tf.float32)
        
    
    print("U.shape", U.shape)
    print("W.shape", W.shape)
    
    #UWfilt = tf.nn.conv1d(U[:,:,np.newaxis],W[:,:,np.newaxis],
    #                      stride=[1],padding='SAME')
    UWfilt = U*W
    #UWfilt = tf.squeeze(UWfilt)
    print("UWfilt.shape", UWfilt.shape)
    
    UW_numpy = UWfilt.numpy()
    plt.figure(231)
    #plt.plot(U.numpy()[0:12:3,:].T, alpha=0.3)
    plt.imshow(U.numpy()[:,:,0,0], alpha=0.3)
    #plt.plot(W)
    plt.figure(232)
    #plt.plot(U.numpy()[0:12:3,:].T, alpha=0.3)
    #plt.plot(UW.numpy()[:,0:-1:1],alpha=0.5)
    plt.subplot(1,len(sis),k)
    plt.imshow(UWfilt.numpy()[:,:,0,0],alpha=0.5)
    
    
    plt.figure(233)
    #plt.plot(U.numpy()[0:12:3,:].T, alpha=0.3)
    plt.subplot(1,len(sis),k)
    plt.imshow(np.var(UWfilt.numpy(),axis=(2,3)), alpha=0.5)
    #plt.colorbar()
    plt.legend(sis)
    #plt.ylim([0,2.0])
    plt.title('Variance')
    
    if wdist!='delta':
        varlistUW.append(np.mean(np.var(UWfilt.numpy(),axis=(2,3))))
        varlistW.append(np.mean(np.var(W.numpy(),axis=(2,3))))
    else:
        varlistUW.append(np.max(np.var(UWfilt.numpy(),axis=(2,3))))
        varlistW.append(np.max(np.var(W.numpy(),axis=(2,3))))
    print("Mean for ", s, ": ", varlistUW[-1]," -W: ",varlistW[-1])
    
    k +=1



import matplotlib as mpl
def paper_fig_settings(addtosize=0):
    #mpl.style.use('seaborn-white')
    mpl.rc('figure',dpi=144)
    mpl.rc('text', usetex=False)
    mpl.rc('axes',titlesize=16+addtosize)
    mpl.rc('xtick', labelsize=12+addtosize)
    mpl.rc('ytick', labelsize=12+addtosize)
    mpl.rc('axes', labelsize=14+addtosize)
    mpl.rc('legend', fontsize=10+addtosize)
    mpl.rc('image',interpolation=None)
    
    
paper_fig_settings(4)
plt.figure(figsize=(12,6))
plt.plot(sis,varlistUW, linewidth=3.0)
plt.plot(sis,varlistW, linewidth=3.0)
#plt.plot(sis,sis**math.sqrt(2.0))
plt.legend(['var[UW]','var[W]'])
#plt.title("Variance of UW vs W in forward mode")
plt.xlabel("$\sigma_s$")
plt.ylabel("Variance")
plt.xlim([0,2.0])
plt.ylim([0.0006,0.0008])
plt.grid(True)

#plt.plot(sis,0.5/(0.5+np.exp(-sis**45))-0.5)

### BACKWARD GRADIENT EXPLORATION MODE







# This part is unnecessary to fit a function for variance degradationg. 
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score




np.random.seed(0)

n_samples = 30
degrees = [2, 3, 4,5]

X = sis
y = np.array(varlistUW)

plt.figure(figsize=(14, 8))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                             scoring="neg_mean_squared_error", cv=10)

    #X_test = np.linspace(0, 1, 100)
    plt.plot(sis, pipeline.predict(X[:, np.newaxis]), label="Fitted")
    plt.plot(sis, varlistUW, label="Value ")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((0, 2))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))
plt.show()

polynomial_features = PolynomialFeatures(degree=4,
                                             include_bias=False)
linear_regression = LinearRegression()
pipeline = Pipeline([("polynomial_features", polynomial_features),
                     ("linear_regression", linear_regression)])
pipeline.fit(X[:, np.newaxis], y)
X_test = np.linspace(0, 5, 100)
plt.plot(sis, pipeline.predict(X[:, np.newaxis]), label="Fitted")
plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Fitted")
plt.plot(sis, varlistUW, label="Value ")
plt.show()


Xexp = np.c_[1./1-np.exp(-sis/2.0),sis**2]
y = np.array(varlistUW)
linear_regression = LinearRegression()
linear_regression.fit(Xexp,y)
plt.figure()
plt.plot(sis,y)
plt.plot(sis,linear_regression.predict(Xexp))
plt.plot(sis,pipeline.predict(X[:, np.newaxis]))
xx = np.c_[sis**1,sis**2, sis**3, sis**4, np.ones(len(sis))]
c = np.array([ 0.49,  2.08261273, -2.51239034,  0.79320175,-0.0631832352338566])
fc = np.dot(xx,c)
plt.plot(sis,fc,'--')
plt.legend(['real','exp', 'poly'])


import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, b, c,d):
    #return c-d*(np.exp(-((x-a)**2)/(d**2) ))+e*x**2+f*x**3
    #return c-d*math.pi*(np.exp(-((x-a)**2)/(b) ))
    bas = d/(math.sqrt(2*math.pi))
    return c-bas*(np.exp(-((x-a)**2)/(b**2) ))

def func_emd(x, a, b, c,d,e,f):
    #return c-d*(np.exp(-((x-a)**2)/(d**2) ))+e*x**2+f*x**3
    #return c-d*math.pi*(np.exp(-((x-a)**2)/(b) ))
    k = np.array([5.69576177e-05, 5.70944988e-01, 9.29715380e-01, 2.26213386e+00])
    bas = k[3]/(math.sqrt(2*math.pi))
    return k[2]-bas*(np.exp(-((x-k[0])**2)/(k[1]**2) ))
# xdata = np.linspace(0.001, 4, 50)
# y = func(xdata, 1.0, 1.0, 0.5,0.0)
#plt.plot(xdata,y)
popt, pcov = curve_fit(func, sis, np.array(varlistUW))
popt

plt.figure()
plt.plot(sis, np.array(varlistUW),'r.')
plt.plot(sis, func(sis, *popt)+0.1, 'b-',alpha=0.5)
plt.grid()
plt.show()
plt.figure()
plt.plot(sis,func(sis, *popt)+0.1-np.array(varlistUW))
plt.show()

plt.figure()
stest= np.linspace(0,2,400)
plt.plot(stest,1.0/(func(stest, *popt)+0.1))
plt.show()

'''              