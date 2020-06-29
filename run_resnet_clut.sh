export CUDA_VISIBLE_DEVICES=1
export TF_FORCE_GPU_ALLOW_GROWTH="true"
#kwargs = {'exname':'resnet', 'dset':'fashion', 'depth':1, 
#              'test_layer':'aconv',
#              'epochs':20, 'batch':128, 
#              'repeats':1, 'kernel_size':7, 'nfilters':16, 
#              'data_augmentation':False, 'lr_multiplier':4.0,
#              'init_sigma':0.15, 'norm':0}

DSET=mnist-clut
EXNAME=resnet_clut
EPOCH=50
BATCH=128
REPEATS=5
DEPTH=1
LRX=1.0
AUGMENT=False
NFILTERS=16

python resnet-acnn-test_trials.py $EXNAME $DSET $DEPTH conv $EPOCH $BATCH $REPEATS 3 $NFILTERS $AUGMENT $LRX;
python resnet-acnn-test_trials.py $EXNAME $DSET $DEPTH conv $EPOCH $BATCH $REPEATS 5 $NFILTERS $AUGMENT $LRX;
python resnet-acnn-test_trials.py $EXNAME $DSET $DEPTH conv $EPOCH $BATCH $REPEATS 7 $NFILTERS $AUGMENT $LRX;
python resnet-acnn-test_trials.py $EXNAME $DSET $DEPTH aconv $EPOCH $BATCH $REPEATS 3 $NFILTERS $AUGMENT $LRX;
python resnet-acnn-test_trials.py $EXNAME $DSET $DEPTH aconv $EPOCH $BATCH $REPEATS 5 $NFILTERS $AUGMENT $LRX;
python resnet-acnn-test_trials.py $EXNAME $DSET $DEPTH aconv $EPOCH $BATCH $REPEATS 7 $NFILTERS $AUGMENT $LRX;



