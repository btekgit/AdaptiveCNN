#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:18:40 2019

@author: btek
"""
import numpy as np
import matplotlib.pyplot as plt
import plot_utils as pu
pu.paper_fig_settings(addtosize=8)
import os
#focus_list = ['outputs/simple_mnist_aconv_5x5_20191130-130151_._results.npz',
#              'outputs/simple_mnist_aconv_7x7_20191130-195607_._results.npz',
#              'outputs/simple_mnist_aconv_9x9_20191130-161328_._results.npz',
#              'outputs/simple_mnist_conv_3x3_20191130-124053_._results.npz',
#              'outputs/simple_mnist_conv_5x5_20191130-203550_._results.npz',
#              'outputs/simple_mnist_conv_7x7_20191130-154824_._results.npz',
#              'outputs/simple_mnist_conv_9x9_20191130-215731_._results.npz']

def get_file_list(folder):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(folder):
        for file in sorted(f):
            if '.npz' in file:
                files.append(os.path.join(r, file))
            
    return files


def plot_results(in_list, ylims=[0,1.0], xlims=None, name='val_acc'):
    N_repeat = 5
    focus_score_list=[]
    #ax = plt.axes(xscale='log')
    #ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    list_to_array = lambda a: [np.array(x) for x in a]
    
    fig = plt.figure(figsize=(8,6))
    
    ax = fig.gca()
    #cmap1 = plt.cm.magma
    #cmap2 = plt.cm.winter
    #cmaps = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    #         '#9467bd', '#8c564b', '#e377c2', '#3f0f0f',
    #          '#bcbd22', '#17becf','#000f0a']
    cmaps = {'conv_7x7':'b', 'conv_5x5':'g', 'aconv_5x5':'r', 'aconv_3x3':'c', 
             'conv_3x3':'m', 'aconv_9x9':'#8c564b', 
             'conv_9x9':'y','aconv_7x7':'k'}
    leg =[]
    
    mxcol =-1.0
    for ii,df in enumerate(in_list):
        ex_name = df.split('/')[-1]
        #print(ex_name)
        fsize= int(ex_name[ex_name.find('x')+1])
        
        if fsize>mxcol:
            mxcol=float(fsize)
    #print(mxcol)
    name_list = []
    scores_list = []
    max_scores_list = []
    for ii,df in enumerate(in_list):
        ex_name = df.split('/')[-1]
        print(ex_name)
        mx_scores = np.load(df)['mx_scores']
        res =  np.load(df,allow_pickle=True)['results'][1]
        
        
        mean_mx_scores = np.mean(mx_scores)
        std_mx_scores =np.std(mx_scores)
        mx_mx_scores = np.max(mx_scores)
        #mean_scores = np.mean(np.reshape(mx_scores,(-1,N_repeat)),axis=1)
        scores_list.append(mx_scores)
        max_scores_list.append([mean_mx_scores,std_mx_scores,mx_mx_scores])
        print("Mean Scores: ",mean_mx_scores," Std Scores: ",std_mx_scores," Max Scores: ",mx_mx_scores)
        
        
        val_acc = np.array([ v[name] for v in res])
        mx = np.max(val_acc,axis=0)
        mn = np.mean(val_acc,axis=0)
        st_p = np.max(val_acc,axis=0)
        st_n = np.min(val_acc,axis=0)
        
        
        
        fsize= int(ex_name[ex_name.find('x')+1])
        le = ex_name[ex_name.find('aconv'):ex_name.find('x')+2]
        
        if le=='':
            le = ex_name[ex_name.find('conv'):ex_name.find('x')+2]
        name_list.append(le)
        if ex_name.find('aconv')>0:
            col = cmaps[le]
            plt.plot(mn,linewidth=2.0, c=col)
        else:
            col = cmaps[le]
            plt.plot(mn,linewidth=2.0,linestyle='--',c=col)
                
        plt.fill_between(np.linspace(0,mn.shape[0],mn.shape[0]),y1=st_n,y2=st_p, alpha=0.1) 
        
        
        leg.append(le)
        # record base values (no-reg) for t-tests
    plt.legend(leg,framealpha=0.2)
    plt.grid('on')
    plt.ylim(ylims)
    plt.xlim(xlims)
    plt.xlabel('Epoch')
    name = name.capitalize()
    print(name)
    if name=='Val_acc':
        name=name.replace('Val_acc','Validation accuracy')
    elif name=='Val_accuracy':
        name=name.replace('Val_','Validation ')
    plt.ylabel(name)
    plt.show()
    
    print(np.array(scores_list))
    for n,s in zip(name_list,scores_list):
        print(n,":",s)
    ar = np.array(max_scores_list)
    #print(ar)
    print("Max Max score:",name_list[np.argmax(ar[:,2])],np.max(ar[:,2]))
    print("Max Mean score:",name_list[np.argmax(ar[:,0])],np.max(ar[:,0]))
    
    return name_list, scores_list
    
def t_test__patterns(name_list,scores_list, patterns=[]):
    
    from scipy.stats import ttest_ind
    # for each pair in patterns it calculates t_tests
    # pattern is an array of string pairs.
    # p[0] is the first name, p[1] is the second name
    
    for p in patterns:
        i1 = name_list.index(p[0])
        i2 = name_list.index(p[1])
        
        print("T-test for pair:", p[0], ":", p[1], ttest_ind(scores_list[i1], scores_list[i2]))
        print("Stats for pair mx, mn,std", np.max(scores_list[i1]),np.mean(scores_list[i1]), 
              np.std(scores_list[i1]),"\n",
              np.max(scores_list[i2]),
              np.mean(scores_list[i2]),
              np.std(scores_list[i2]))

# In[5]:

mnist_list=get_file_list('outputs/lr01_mnist')
             
plot_results(mnist_list,[0.99,0.997])
# cifar-10
cif_list = get_file_list('outputs/lr01_cifar')
plot_results(cif_list,[0.00,0.915])


cif_do_list = get_file_list('outputs/cifar_do_b32_lr01')
plot_results(cif_do_list,[0.0,0.915])


mnist_clut = get_file_list('outputs/lr001_clut')
plot_results(mnist_clut,[0.6,0.999])


fashion_list = get_file_list('outputs/lr01_fashion')
plot_results(fashion_list,[0.90,0.945])

mnist_fixed =mnist_list=get_file_list('outputs/fixed_sigma/mnist')
plot_results(mnist_fixed,[0.99,0.997])

cif_fixed = get_file_list('outputs/fixed_sigma/cifar10')
plot_results(cif_fixed,[0.0,0.915])


resnet_mnist_list = get_file_list('outputs/resnet/mnist')
plot_results(resnet_mnist_list,[0.95,0.998])

resnet_cif_do_list = get_file_list('outputs/resnet/cifar10')
plot_results(resnet_cif_do_list,[0.70,0.935])


resnet_mnist_clut = get_file_list('outputs/resnet/clut')
plot_results(resnet_mnist_clut,[0.95,0.999])


resnet_fashion_list = get_file_list('outputs/resnet/fashion')
plot_results(resnet_fashion_list,[0.85,0.945])

resnet_faces_list = get_file_list('outputs/resnet/faces')
plot_results(resnet_faces_list,[0.85,0.945])




resnet_mnist_list = get_file_list('outputs/resnet/dropout05/mnist2')
plot_results(resnet_mnist_list,[0.97,0.998])

resnet_cif_do_list = get_file_list('outputs/resnet/dropout05/cifar10')
plot_results(resnet_cif_do_list,[0.85,0.935])


resnet_mnist_clut = get_file_list('outputs/resnet/dropout05/clut')
plot_results(resnet_mnist_clut,[0.5,0.999])


resnet_fashion_list = get_file_list('outputs/resnet/dropout05/fashion')
plot_results(resnet_fashion_list,[0.85,0.945])

resnet_faces_list = get_file_list('outputs/resnet/dropout05/faces')
plot_results(resnet_faces_list,[0.20,0.945])

resnet_cif_do_list = get_file_list('outputs/resnet/cifar10_large')
plot_results(resnet_cif_do_list,[0.50,0.935])

# In[may2020]:

def add_to_cumplot(scr_list, i,m ,xlims=[0.9,1.0],ylims=[0.9,1.0],lab=None):
    f = plt.figure(633)
    ax = f.gca()
    ax.plot([0.7,1.0],[0.7,1.0],'--', alpha=0.5)  
    ax.plot(np.array(scr_list)[1:i],np.array(scr_list)[i+1:],m, alpha=0.5,label=lab)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    f.legend()
    



mnist_list = get_file_list('outputs/may2020/mnist')
nm_list, scr_list = plot_results(mnist_list,[0.987,0.996])
t_test__patterns(nm_list,scr_list,[('aconv_7x7','conv_7x7')])
plot_results(mnist_list,ylims=[0.,0.02],name='loss')
plot_results(mnist_list,ylims=[0.015,0.03],name='val_loss')
add_to_cumplot(scr_list,4,'r*',xlims=[0.75,1.0],ylims=[0.75,1.0],lab='mnist')

cifar_list = get_file_list('outputs/may2020/cifar10')
nm_list, scr_list = plot_results(cifar_list,[0.74,0.795])
t_test__patterns(nm_list,scr_list,[('aconv_9x9','conv_9x9')])
plot_results(cifar_list,ylims=[0.3,0.6],name='loss')
plot_results(cifar_list,ylims=[0.6,1.0],name='val_loss')
add_to_cumplot(scr_list,4,'b+',xlims=[0.75,1.0],ylims=[0.75,1.0],lab='cifar')


fashion_list = get_file_list('outputs/may2020/fashion')
nm_list, scr_list =plot_results(fashion_list,[0.925,0.941])
t_test__patterns(nm_list,scr_list,[('aconv_9x9','conv_9x9')])
plot_results(fashion_list,ylims=[0.09,0.3],name='loss')
add_to_cumplot(scr_list,4,'cp',xlims=[0.75,1.0],ylims=[0.75,1.0],lab='fashion')

clut_list = get_file_list('outputs/may2020/mnist-clut')
nm_list, scr_list =plot_results(clut_list,[0.82,0.96])
t_test__patterns(nm_list,scr_list,[('aconv_9x9','conv_9x9')])
plot_results(clut_list,ylims=[0.00,0.2],name='loss')
add_to_cumplot(scr_list,4,'gs',xlims=[0.75,1.0],ylims=[0.75,1.0],lab='clut')

faces_list = get_file_list('outputs/may2020/faces/lr001/old')
nm_list, scr_list = plot_results(faces_list,[0.75,0.85])
t_test__patterns(nm_list,scr_list,[('aconv_9x9','conv_9x9')])
plot_results(faces_list,ylims=[0.0,0.3],name='loss')
plot_results(faces_list,ylims=[0.0,1.5],name='val_loss')
add_to_cumplot(scr_list,4,'rd',xlims=[0.75,1.0],ylims=[0.75,1.0],lab='faces')

#faces_list = get_file_list('outputs/may2020/faces/lr001/sigma_decay_slower')
#plot_results(faces_list,[0.75,0.86])




resnet_mnist_list = get_file_list('outputs/may2020/resnet/mnist')
nm_list, scr_list = plot_results(resnet_mnist_list,[0.9850,0.998])
mx_row = np.unravel_index(np.argmax(scr_list, axis=None), np.array(scr_list).shape)[0]
mx_name = nm_list[mx_row]
t_test__patterns(nm_list,scr_list,[('aconv_5x5','conv_5x5')])
add_to_cumplot(scr_list,3,'r*',lab='mnist')

resnet_clut_list = get_file_list('outputs/may2020/resnet/clut')
nm_list, scr_list =plot_results(resnet_clut_list,[0.9,0.998])
mx_row = np.unravel_index(np.argmax(scr_list, axis=None), np.array(scr_list).shape)[0]
mx_name = nm_list[mx_row]
t_test__patterns(nm_list,scr_list,[('aconv_7x7','conv_7x7')])
add_to_cumplot(scr_list,3,'gs',lab='clut')
#resnet_fashion_list = get_file_list('outputs/may2020/resnet/fashion')
#plot_results(resnet_fashion_list,[0.80,0.95])

resnet_fashion_list =resnet_fashion_list = get_file_list('outputs/may2020/resnet/fashion/lr_01')
nm_list, scr_list=plot_results(resnet_fashion_list,[0.80,0.95])
mx_row = np.unravel_index(np.argmax(scr_list, axis=None), np.array(scr_list).shape)[0]
mx_name = nm_list[mx_row]
t_test__patterns(nm_list,scr_list,[('aconv_7x7','conv_7x7')])
add_to_cumplot(scr_list,3,'bs',lab='fashion')

#resnet_cifar_list = get_file_list('outputs/may2020/resnet/cifar10_2')
#plot_results(resnet_cifar_list,[0.59,0.93])

resnet_cifar_list = get_file_list('outputs/may2020/resnet/cifar10_lr01')
nm_list, scr_list=plot_results(resnet_cifar_list,[0.7,0.93])
mx_row = np.unravel_index(np.argmax(scr_list, axis=None), np.array(scr_list).shape)[0]
mx_name = nm_list[mx_row]
t_test__patterns(nm_list,scr_list,[('aconv_5x5','conv_5x5')])
add_to_cumplot(scr_list,3,'b>',lab='cifar10')

resnet_faces_list = get_file_list('outputs/may2020/resnet/faces/paper')
nm_list, scr_list=plot_results(resnet_faces_list,[0.70,0.98])
mx_row = np.unravel_index(np.argmax(scr_list, axis=None), np.array(scr_list).shape)[0]
mx_name = nm_list[mx_row]
nm_list, scr_list=plot_results(resnet_faces_list,ylims=None,name='loss')
nm_list, scr_list=plot_results(resnet_faces_list,ylims=None,name='val_loss')
t_test__patterns(nm_list,scr_list,[('aconv_3x3','conv_3x3')])
add_to_cumplot(scr_list,3,'r^',lab='faces')


#In[]:
#resnet_faces_list = get_file_list('outputs/may2020/resnet/faces/drp_out')
#nm_list, scr_list=plot_results(resnet_faces_list,[0.80,0.98])
#adam result
resnet_faces_list = get_file_list('/home/btek/Dropbox/code/pythoncode/AdaptiveCNN/outputs/aug2020/resnet')
nm_list, scr_list=plot_results(resnet_faces_list,[0.80,0.98])
nm_list, scr_list=plot_results(resnet_faces_list,ylims=None,name='loss')
    

# In[aug2020]:

def add_to_cumplot(scr_list, i, m ,xlims=[0.9,1.0],ylims=[0.9,1.0],lab=None):
    f = plt.figure(633)
    ax = f.gca()
    ax.plot([0.5,1.0],[0.5,1.0],'--', alpha=0.5)  
    ax.plot(np.array(scr_list)[1:i], np.array(scr_list)[i+1:], m, alpha=0.5,label=lab)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    f.legend()
    



mnist_list = get_file_list('outputs/aug2020/mnist')
nm_list, scr_list = plot_results(mnist_list,[0.990,0.9965],name='val_accuracy')
mx_row = np.unravel_index(np.argmax(scr_list, axis=None), np.array(scr_list).shape)[0]
mx_name = nm_list[mx_row]
t_test__patterns(nm_list,scr_list,[('aconv_9x9','conv_9x9')])
plot_results(mnist_list,ylims=[0.,0.02],name='loss')
plot_results(mnist_list,ylims=[0.015,0.03],name='val_loss')
add_to_cumplot(scr_list,4,'r*',xlims=[0.75,1.0],ylims=[0.75,1.0],lab='mnist')

cifar_list = get_file_list('outputs/aug2020/cifarlr01')
nm_list, scr_list = plot_results(cifar_list,[0.75,0.8],name='val_accuracy')
mx_row = np.unravel_index(np.argmax(scr_list, axis=None), np.array(scr_list).shape)[0]
mx_name = nm_list[mx_row]
t_test__patterns(nm_list,scr_list,[('aconv_9x9','conv_9x9')])
plot_results(cifar_list,ylims=[0.1,0.6],name='loss')
plot_results(cifar_list,ylims=[0.6,1.0],name='val_loss')
add_to_cumplot(scr_list,4,'b+',xlims=[0.75,1.0],ylims=[0.75,1.0],lab='cifar')


fashion_list = get_file_list('outputs/aug2020/fashion')
nm_list, scr_list =plot_results(fashion_list,[0.925,0.941],name='val_accuracy')
mx_row = np.unravel_index(np.argmax(scr_list, axis=None), np.array(scr_list).shape)[0]
mx_name = nm_list[mx_row]
t_test__patterns(nm_list,scr_list,[('aconv_7x7','conv_7x7')])
plot_results(fashion_list,ylims=[0.01,0.3],name='loss')
add_to_cumplot(scr_list,4,'cp',xlims=[0.75,1.0],ylims=[0.75,1.0],lab='fashion')

clut_list = get_file_list('outputs/aug2020/mnist_clut')
nm_list, scr_list =plot_results(clut_list,[0.82,0.96],name='val_accuracy')
mx_row = np.unravel_index(np.argmax(scr_list, axis=None), np.array(scr_list).shape)[0]
mx_name = nm_list[mx_row]
t_test__patterns(nm_list,scr_list,[('aconv_9x9','conv_9x9')])
plot_results(clut_list,ylims=[0.00,0.2],name='loss')
add_to_cumplot(scr_list,4,'gs',xlims=[0.75,1.0],ylims=[0.75,1.0],lab='clut')

faces_list = get_file_list('outputs/aug2020/faces/')
nm_list, scr_list = plot_results(faces_list,[0.7,0.87],name='val_accuracy')
mx_row = np.unravel_index(np.argmax(scr_list, axis=None), np.array(scr_list).shape)[0]
mx_name = nm_list[mx_row]
t_test__patterns(nm_list,scr_list,[('aconv_9x9','conv_9x9')])
plot_results(faces_list,ylims=[.0,2.9],name='loss')
plot_results(faces_list,ylims=[0.0,1.5],name='val_loss')
add_to_cumplot(scr_list,4,'rd',xlims=[0.75,1.0],ylims=[0.75,1.0],lab='faces')

plt.show()
#faces_list = get_file_list('outputs/may2020/faces/lr001/sigma_decay_slower')
#plot_results(faces_list,[0.75,0.86])
