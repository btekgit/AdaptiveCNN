export CUDA_VISIBLE_DEVICES=1
export TF_FORCE_GPU_ALLOW_GROWTH="true"
#{'dset':'mnist', 'arch':'simple', 'repeats':1, 'test_layer':'aconv',
#              'epochs':10, 'batch':128, 'exname':'noname', 
#              'adaptive_kernel_size':5, 'nfilters':32, 'data_augmentation':True}

SNAME=../Conv2DAdaptive_k2.py

python $SNAME mnist-clut simple 5 conv 100 128 tf2 3 32 False 0.1;
python $SNAME mnist-clut simple 5 conv 100 128 tf2 5 32 False 0.1;
python $SNAME mnist-clut simple 5 conv 100 128 tf2 7 32 False 0.1;
python $SNAME mnist-clut simple 5 conv 100 128 tf2 9 32 False 0.1;
python $SNAME mnist-clut simple 5 aconv 100 128 tf2 3 32 False 0.1;
python $SNAME mnist-clut simple 5 aconv 100 128 tf2 5 32 False 0.1;
python $SNAME mnist-clut simple 5 aconv 100 128 tf2 7 32 False 0.1;
python $SNAME mnist-clut simple 5 aconv 100 128 tf2 9 32 False 0.1;

