export CUDA_VISIBLE_DEVICES=1
export TF_FORCE_GPU_ALLOW_GROWTH="true"
#kwargs = {'exname':'resnet', 'dset':'fashion', 'depth':1, 
#              'test_layer':'aconv',
#              'epochs':20, 'batch':128, 
#              'repeats':1, 'kernel_size':7, 'nfilters':16, 
#              'data_augmentation':False, 'lr_multiplier':4.0,
#              'init_sigma':0.15, 'norm':0}

DSET=fashion
EXNAME=resnet_fashion
EPOCH=100
BATCH=128
REPEATS=5
DEPTH=2
LRX=0.1
AUGMENT=True
NFILTERS=16

SNAME=../resnet_acnn_test_trials.py

python $SNAME $EXNAME $DSET $DEPTH aconv $EPOCH $BATCH $REPEATS 7 $NFILTERS $AUGMENT $LRX;
python $SNAME $EXNAME $DSET $DEPTH aconv $EPOCH $BATCH $REPEATS 3 $NFILTERS $AUGMENT $LRX;
python $SNAME $EXNAME $DSET $DEPTH aconv $EPOCH $BATCH $REPEATS 5 $NFILTERS $AUGMENT $LRX;
python $SNAME $EXNAME $DSET $DEPTH conv $EPOCH $BATCH $REPEATS 3 $NFILTERS $AUGMENT $LRX;
python $SNAME $EXNAME $DSET $DEPTH conv $EPOCH $BATCH $REPEATS 5 $NFILTERS $AUGMENT $LRX;
python $SNAME $EXNAME $DSET $DEPTH conv $EPOCH $BATCH $REPEATS 7 $NFILTERS $AUGMENT $LRX;



