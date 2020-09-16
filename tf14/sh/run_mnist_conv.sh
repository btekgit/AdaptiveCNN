export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH="true"
#kwargs = {'dset':'mnist', 'arch':'simple', 'repeats':1, 
#              'test_layer':'aconv',
#              'epochs':100, 'batch':128, 'exname':'noname', 
#              'adaptive_kernel_size':7, 'nfilters':32, 
#              'data_augmentation':False, 'lr_multiplier':1.0}
    # For MNIST YOU CAN USE lr_multiplier 1.0, 
    
SNAME=../Conv2DAdaptive_k2.py

python $SNAME mnist simple 5 conv 100 128 tf2 3 32 False;
python $SNAME mnist simple 5 conv 100 128 tf2 5 32 False;
python $SNAME mnist simple 5 conv 100 128 tf2 7 32 False;
python $SNAME mnist simple 5 conv 100 128 tf2 9 32 False;
python $SNAME mnist simple 5 aconv 100 128 tf2 3 32 False;
python $SNAME mnist simple 5 aconv 100 128 tf2 5 32 False;
python $SNAME mnist simple 5 aconv 100 128 tf2 7 32 False;
python $SNAME mnist simple 5 aconv 100 128 tf2 9 32 False;


