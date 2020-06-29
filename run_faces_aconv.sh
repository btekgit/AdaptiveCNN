export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH="true"

#{'dset':'mnist', 'arch':'simple', 'repeats':1, 'test_layer':'aconv',
#              'epochs':10, 'batch':32, 'exname':'noname', 
#              'adaptive_kernel_size':5, 'nfilters':32, 'data_augmentation':$AUGMENT}

DSET=lfw_faces
EXNAME=lr001_b8_n32
EPOCH=100
BATCH=8
REPEATS=5
LRX=0.01
NFILTERS=32
AUGMENT=False

python Conv2DAdaptive_k.py $DSET simple $REPEATS aconv $EPOCH $BATCH $EXNAME 3 $NFILTERS $AUGMENT $LRX;
python Conv2DAdaptive_k.py $DSET simple $REPEATS aconv $EPOCH $BATCH $EXNAME 5 $NFILTERS $AUGMENT $LRX;
python Conv2DAdaptive_k.py $DSET simple $REPEATS aconv $EPOCH $BATCH $EXNAME 7 $NFILTERS $AUGMENT $LRX;
python Conv2DAdaptive_k.py $DSET simple $REPEATS aconv $EPOCH $BATCH $EXNAME 9 $NFILTERS $AUGMENT $LRX;
python Conv2DAdaptive_k.py $DSET simple $REPEATS conv $EPOCH $BATCH $EXNAME 3 $NFILTERS $AUGMENT $LRX;
python Conv2DAdaptive_k.py $DSET simple $REPEATS conv $EPOCH $BATCH $EXNAME 5 $NFILTERS $AUGMENT $LRX;
python Conv2DAdaptive_k.py $DSET simple $REPEATS conv $EPOCH $BATCH $EXNAME 7 $NFILTERS $AUGMENT $LRX;
python Conv2DAdaptive_k.py $DSET simple $REPEATS conv $EPOCH $BATCH $EXNAME 9 $NFILTERS $AUGMENT $LRX;



