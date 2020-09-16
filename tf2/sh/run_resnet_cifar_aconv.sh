export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH="true"
#kwargs = {'exname':'resnet', 'dset':'fashion', 'depth':1, 
#              'test_layer':'aconv',
#              'epochs':20, 'batch':128, 
#              'repeats':1, 'kernel_size':7, 'nfilters':16, 
#              'data_augmentation':False, 'lr_multiplier':4.0,
#              'init_sigma':0.15, 'norm':0}

EXNAME=resnet_cifar10
EPOCH=200
BATCH=128
REPEATS=5
DEPTH=3
DSET=cifar10
LRX=0.1
AUGMENT=True
NFILTERS=16
SNAME=../resnet_acnn_test_trials_tf2.py


python $SNAME $EXNAME $DSET $DEPTH aconv $EPOCH $BATCH $REPEATS 7 $NFILTERS $AUGMENT $LRX;
python $SNAME $EXNAME $DSET $DEPTH aconv $EPOCH $BATCH $REPEATS 3 $NFILTERS $AUGMENT $LRX;
python $SNAME $EXNAME $DSET $DEPTH aconv $EPOCH $BATCH $REPEATS 5 $NFILTERS $AUGMENT $LRX;
python $SNAME $EXNAME $DSET $DEPTH conv $EPOCH $BATCH $REPEATS 3 $NFILTERS $AUGMENT $LRX;
python $SNAME $EXNAME $DSET $DEPTH conv $EPOCH $BATCH $REPEATS 5 $NFILTERS $AUGMENT $LRX;
python $SNAME $EXNAME $DSET $DEPTH conv $EPOCH $BATCH $REPEATS 7 $NFILTERS $AUGMENT $LRX;
#python resnet-acnn-test_trials.py $EXNAME $DSET $DEPTH conv $EPOCH $BATCH $REPEATS 15 $NFILTERS $AUGMENT $LRX;
#python resnet-acnn-test_trials.py $EXNAME $DSET $DEPTH aconv $EPOCH $BATCH $REPEATS 15 $NFILTERS $AUGMENT $LRX;
#python resnet-acnn-test_trials.py $EXNAME $DSET $DEPTH aconv $EPOCH $BATCH $REPEATS 21 $NFILTERS $AUGMENT $LRX;



