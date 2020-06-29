export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH="true"
#{'dset':'mnist', 'arch':'simple', 'repeats':1, 'test_layer':'aconv',
#              'epochs':10, 'batch':256, 'exname':'noname', 
#              'adaptive_kernel_size':5, 'nfilters':32, 'data_augmentation':True}
python Conv2DAdaptive_k.py cifar10 simple 5 conv 120 128 noname 3 32 False;
python Conv2DAdaptive_k.py cifar10 simple 5 conv 120 128 noname 5 32 False;
python Conv2DAdaptive_k.py cifar10 simple 5 conv 120 128 noname 7 32 False;
python Conv2DAdaptive_k.py cifar10 simple 5 conv 120 128 noname 9 32 False;
python Conv2DAdaptive_k.py cifar10 simple 5 aconv 120 128 noname 3 32 False;
python Conv2DAdaptive_k.py cifar10 simple 5 aconv 120 128 noname 5 32 False;
python Conv2DAdaptive_k.py cifar10 simple 5 aconv 120 128 noname 7 32 False;
python Conv2DAdaptive_k.py cifar10 simple 5 aconv 120 128 noname 9 32 False;


