# AdaptiveCNN
Repo for Adaptive Convolution Kernel for Artificial Neural Networks

https://arxiv.org/abs/2009.06385

A method for training the size of convolutional kernels to provide varying 
size kernels in a single layer. 
The method utilizes a differentiable, and therefore backpropagation-trainable 
Gaussian envelope which can grow or shrink in a base grid.

First of all some of the experiments were done inm Tensorflow 1.4
Later many of them was translated to Tensorflow 2.0
Unfortunately moving to Tensorflow I had to give up separate learning rates 
for sigma parameter

Therefore there are two folders for tf1.4 and tf2.0
Inside there are scripts for four different experiments. 

1) testfilters.py: Autoencoder experiments
2) Conv2DAdaptive_k2.py: The main model adaptive convolution implementation and simple convolution network experiments
3) resnet_acnn_test_trials_tf2.py: Resnet model experiments
4) test_ox_pets_acnn_tf_data.py: U-net segmentation experiment





