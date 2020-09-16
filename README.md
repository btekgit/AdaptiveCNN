# AdaptiveCNN
### Repo for Adaptive Convolution Kernel for Artificial Neural Networks

https://arxiv.org/abs/2009.06385

A method for training the size of convolutional kernels to provide varying 
size kernels in a single layer. 
The method utilizes a differentiable, and therefore backpropagation-trainable 
Gaussian envelope which can grow or shrink in a base grid.

First of all some of the experiments were done initially in Tensorflow 1.4.
Later many of them was translated to Tensorflow 2.0.
Unfortunately moving to Tensorflow 2.0, we had to give up separate learning rates 
for the sigma parameter.

Therefore there are two folders for tf1.4 and tf2.0
Inside there are scripts for four different experiments. 

1) testfilters.py: Autoencoder experiments (tf2.0 up to date)
2) Conv2DAdaptive_k2.py: The main model adaptive convolution 
implementation and simple convolution network experiments (tf2.0 up to date, at least for mnist, cifar10)
3) resnet_acnn_test_trials_tf2.py: Resnet model experiments (tf2.0 working but paper experiments used tf1.4)
4) test_ox_pets_acnn_tf_data.py: U-net segmentation experiment (tf2.0 only)





