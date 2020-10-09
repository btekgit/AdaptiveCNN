#!/usr/bin/env python
# coding: utf-8

# # Analyzing with iNNvestigate

# **iNNvestigate** got created to make analyzing neural network's predictions easy! The library should help the user to focus on research and development by providing implemented analysis methods and facilitating rapid development of new methods. In this notebook we will show you how to use **iNNvestigate** and for a better understanding we recommend to read [iNNvestigate neural networks!](https://arxiv.org/abs/1808.04260) first! How to use **iNNvestigate** you can read in this notebook: [Developing with iNNvestigate](introduction_development.ipynb)
# 
# -----
# 
# **The intention behind iNNvestigate is to make it easy to use analysis methods, but it is not to explain the underlying concepts and assumptions. Please, read the according publication(s) when using a certain method and when publishing please cite the according paper(s) (as well as the [iNNvestigate paper](https://arxiv.org/abs/1808.04260)). Thank you!** You can find most related publication in [iNNvestigate neural networks!](https://arxiv.org/abs/1808.04260) and in the README file.
# 
# ### Analysis methods
# 
# The field of analyizing neural network's predictions is about gaining insights how and why a potentially complex network gave as output a certain value or choose a certain class over others. This is often called interpretability or explanation of neural networks. We just call it analyzing a neural network's prediction to be as neutral as possible and to leave any conclusions to the user.
# 
# Most methods have in common that they analyze the input features w.r.t. a specific neuron's output. Which insights a method reveals about this output can be grouped into (see [Learning how to explain: PatternNet and PatternAttribution](https://arxiv.org/abs/1705.05598)):
# 
# * **function:** analyzing the operations the network function uses to extract or compute the output. E.g., how would changing an input feature change the output.
# * **signal:** analyzing the components of the input that cause the output. E.g., which parts of an input image or which directions of an input are used to determine the output.
# * **attribution:** attributing the "importance" of input features for the output. E.g., how much would changing an input feature change the output.
# 
# ----
# 
# In this notebook we will introduce methods for each of these categories and along show how to use different features of **iNNvestigate**, namely how to:
# 
# * analyze a prediction.
# * train an analyzer.
# * analyze a prediction w.r.t to a specific output neuron.
# 
# Let's dive right into it!

# 
# ### Training a network
# 
# To analyze a network, we need a network! As a base for **iNNvestigate** we chose the Keras deep learning library, because it is easy to use and allows to inspect build models.
# 
# In this first piece of code we import all the necessary modules:

# In[1]:


import warnings
warnings.simplefilter('ignore')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import imp
import matplotlib.pyplot as plot
import numpy as np
import os

if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys():
    os.environ['CUDA_VISIBLE_DEVICES']="0"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"

import keras
import keras.backend
import keras.layers
import keras.models
import keras.utils

import innvestigate
import innvestigate.utils as iutils

import matplotlib.pyplot as plt

# Use utility libraries to focus on relevant iNNvestigate routines.
mnistutils = imp.load_source("utils_mnist", "../../innvestigate/examples/utils_mnist.py")


# In[ ]:


# Load data
# returns x_train, y_train, x_test, y_test as numpy.ndarray
data_not_preprocessed = mnistutils.fetch_data()

# Create preprocessing functions
input_range = [-1, 1]
preprocess, revert_preprocessing = mnistutils.create_preprocessing_f(data_not_preprocessed[0], input_range)

# Preprocess data
data = (
    preprocess(data_not_preprocessed[0]), data_not_preprocessed[1],
    preprocess(data_not_preprocessed[2]), data_not_preprocessed[3]
)

num_classes = len(np.unique(data[1]))
label_to_class_name = [str(i) for i in range(num_classes)]


# In[3]:


from Conv2DAdaptive_k import test_mnist
from keras_data import load_dataset


# In[4]:


#settings
settings = {'dset':'fashion', 'arch':'simple', 'repeats':1, 
              'test_layer':'aconv',
              'epochs':1, 'batch':128, 'exname':'noname', 
              'adaptive_kernel_size':7, 'nfilters':32, 
              'data_augmentation':False, 'lr_multiplier':1.0}

# load data 
dset = settings['dset']
batch_size = settings['batch']
num_classes = 10
epochs =settings['epochs']
test_acnn = settings['test_layer']=='aconv'
normalize_data = True
if dset=='mnist-clut':
    normalize_data=False

ld_data = load_dataset(dset,normalize_data,options=[])
x_train,y_train,x_test,y_test,input_shape,num_classes=ld_data


# to load the data:


# In[ ]:


if not os.path.exists('outputs'):
    print("creating output directory")
    os.mkdir('outputs')
print("creating output directory")
sc, hs, model, cb = test_mnist(settings,sid=31+19*17)
fname = 'outputs/'+settings['dset']+'_'+settings['test_layer']+'_'+str(
        settings['adaptive_kernel_size'])+'_Epc'+str(
                settings['epochs'])+''

model.save(fname)
load_models = False
if load_models:
    if settings['dset']=='fashion':
        if settings['adaptive_kernel_size']==7:
            if settings['test_layer']=='conv':
                model.load('outputs/fashion_conv_7x7')
            
    if settings['dset']=='mnist':
        if settings['test_layer']=='conv':
            if settings['adaptive_kernel_size']==7:
                model.load('outputs/mnist_conv_7x7')
    
scores = model.evaluate(x_test, y_test, batch_size=128)
print("Scores on test set: loss=%s accuracy=%s" % tuple(scores[0:2]))


# ## Analyzing a predicition
# 
# Let's first choose an image to analyze:

# In[ ]:

# lets get correct and misclassidied
preds = model.predict(x_test)
pred_scores_ix = np.argsort(np.max(preds,axis=1))
selected_examples = np.concatenate([pred_scores_ix[0:5], pred_scores_ix[-5:]],axis=0)


# In[ ]:


# Choosing a test image for the tutorial:
image = x_test[7:8]

plot.imshow(image.squeeze(), cmap='gray', interpolation='nearest')
plot.show()


# In this first part we show how to create and use an analyzer. To do so we use an analyzer from *function* category, namely the gradient. The gradient shows how the linearized network function reacts on changes of a single feature.
# 
# This is simply done by passing the model without a softmax to the analyzer class:

# In[ ]:


# Stripping the softmax activation from the model
model_wo_sm = iutils.keras.graph.model_wo_softmax(model)

# Creating an analyzer
gradient_analyzer = innvestigate.analyzer.Gradient(model_wo_sm)

# Applying the analyzer
analysis = gradient_analyzer.analyze(image)

# Displaying the gradient
plot.imshow(analysis.squeeze(), cmap='seismic', interpolation='nearest')
plot.show()


# For convience there is a function that creates an analyzer for you. It passes all the parameter on to the class instantiation:

# In[ ]:


# Creating an analyzer
gradient_analyzer = innvestigate.create_analyzer("gradient", model_wo_sm)

# Applying the analyzer
analysis = gradient_analyzer.analyze(image)

# Displaying the gradient
plot.imshow(analysis.squeeze(), cmap='seismic', interpolation='nearest')
plot.show()



# To emphasize different compontents of the analysis many people use instead of the "plain" gradient the absolute value or the square of it. With the gradient analyzer this can be done specifying additional parameters when creating the analyzer:

# In[ ]:


# Creating a parameterized analyzer
abs_gradient_analyzer = innvestigate.create_analyzer("gradient", model_wo_sm, postprocess="abs")
square_gradient_analyzer = innvestigate.create_analyzer("gradient", model_wo_sm, postprocess="square")


# Similar other analyzers can be parameterized.
# 
# Now we visualize the result by projecting the gradient into a gray-color-image:

# In[ ]:


# Applying the analyzers
abs_analysis = abs_gradient_analyzer.analyze(image)
square_analysis = square_gradient_analyzer.analyze(image)

# Displaying the analyses, use gray map as there no negative values anymore
plot.imshow(abs_analysis.squeeze(), cmap='gray', interpolation='nearest')
plot.show()
plot.imshow(square_analysis.squeeze(), cmap='gray', interpolation='nearest')
plot.show()


# ## Training an analyzer
# 
# Some analyzers are data-dependent and need to be trained. In **iNNvestigate** this realized with a SKLearn-like interface. In the next piece of code we train the method PatternNet that analyzes the *signal*:

# In[ ]:


## Creating an analyzer
#patternnet_analyzer = innvestigate.create_analyzer("pattern.net", model_wo_sm, pattern_type="relu")
#
## Train (or adapt) the analyzer to the training data
#patternnet_analyzer.fit(x_train[0], verbose=True)
#
## Applying the analyzer
#analysis = patternnet_analyzer.analyze(image)
#
#
## And visualize it:
#
## In[ ]:
#
#
## Displaying the signal (projected back into input space)
#plot.imshow(analysis.squeeze()/np.abs(analysis).max(), cmap="gray", interpolation="nearest")
#plot.show()


# ## Choosing the output neuron
# 
# In the previous examples we always analyzed the output of the neuron with the highest activation. In the next one we show how one can choose the neuron to analyze:

# In[ ]:


# Creating an analyzer and set neuron_selection_mode to "index"
inputXgradient_analyzer = innvestigate.create_analyzer("input_t_gradient", model_wo_sm,
                                                       neuron_selection_mode="index")


# The gradient\*input analyzer is an example from the *attribution* category and we visualize it by means of a colored heatmap to highlight positive and negative attributions:

# In[ ]:


for neuron_index in range(10):
    print("Analysis w.r.t. to neuron", neuron_index)
    # Applying the analyzer and pass that we want 
    analysis = inputXgradient_analyzer.analyze(image, neuron_index)
    
    # Displaying the gradient
    plot.imshow(analysis.squeeze(), cmap='seismic', interpolation='nearest')
    plot.show()





# In[ ]:
    
# ## Additional resources
# 
# If you would like to learn more we have more notebooks for you, 
# for example: [Comparing methods on MNIST]
# (mnist_method_comparison.ipynb), 
# [Comparing methods on ImageNet]
# (imagenet_method_comparison.ipynb) and 
# [Comparing networks on ImageNet](imagenet_network_comparison.ipynb)
# 
# If you want to know more about how to use the API of **iNNvestigate** look into: [Developing with iNNvestigate](introduction_development.ipynb)

# Scale to [0, 1] range for plotting.
def input_postprocessing(X):
    return revert_preprocessing(X) / 255

noise_scale = (input_range[1]-input_range[0]) * 0.1
ri = input_range[0]  # reference input


# Configure analysis methods and properties
methods = [
    # NAME                    OPT.PARAMS                POSTPROC FXN               TITLE

    # Show input
    ("input",                 {},                       input_postprocessing,      "Input"),

    # Function
    #("gradient",              {"postprocess": "abs"},   mnistutils.graymap,        "Gradient"),
    #("smoothgrad",            {"noise_scale": noise_scale,
    #                           "postprocess": "square"},mnistutils.graymap,        "SmoothGrad"),

    # Signal
    #("deconvnet",             {},                       mnistutils.bk_proj,        "Deconvnet"),
    #("guided_backprop",       {},                       mnistutils.bk_proj,        "Guided Backprop",),
    #("pattern.net",           {"pattern_type": "relu"}, mnistutils.bk_proj,        "PatternNet"),

    # Interaction
    #("pattern.attribution",   {"pattern_type": "relu"}, mnistutils.heatmap,        "PatternAttribution"),
    ("deep_taylor.bounded",   {"low": input_range[0],
                               "high": input_range[1]}, mnistutils.heatmap,        "DeepTaylor"),
    #("input_t_gradient",      {},                       mnistutils.heatmap,        "Input * Gradient"),
    #("integrated_gradients",  {"reference_inputs": ri}, mnistutils.heatmap,        "Integrated Gradients"),
    #("deep_lift.wrapper",     {"reference_inputs": ri}, mnistutils.heatmap,        "DeepLIFT Wrapper - Rescale"),
    #("deep_lift.wrapper",     {"reference_inputs": ri, "nonlinear_mode": "reveal_cancel"},
    #                                                    mnistutils.heatmap,        "DeepLIFT Wrapper - RevealCancel"),
    ("lrp.z",                 {},                       mnistutils.heatmap,        "LRP-Z"),
    #("lrp.epsilon",           {"epsilon": 1},           mnistutils.heatmap,        "LRP-Epsilon"),
    #("lrp.alpha_2_beta_1_IB", {},                       mnistutils.heatmap,        "LRP-A2B1"),
    
]
    
    
    # Create model without trailing softmax
model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)

# Create analyzers.
analyzers = []
for method in methods:
    analyzer = innvestigate.create_analyzer(method[0],        # analysis method identifier
                                            model_wo_softmax, # model without softmax output
                                            **method[1])      # optional analysis parameters

    # Some analyzers require training.
    analyzer.fit(x_train[0], batch_size=256, verbose=1)
    analyzers.append(analyzer)
    
    
# In[]
    
n = 10
preds = model.predict(x_test)
pred_scores_ix = np.argsort(np.max(preds,axis=1))
selected_examples = np.concatenate([pred_scores_ix[0:5], pred_scores_ix[-5:]],axis=0)
#fashion
if settings['dset']=='mnist':
    selected_examples = [1737, 1709, 9634, 1242, 4284, 4452, 4450, 4448, 4468, 9999]
elif settings['dset']=='fashion':
    selected_examples =[6874, 6676, 3229,  332, 6625, 2687, 8169, 8170, 5044, 2916]
    
test_images = list(zip(x_test[selected_examples], np.argmax(y_test[selected_examples],axis=1)))

analysis = np.zeros([len(test_images), len(analyzers), 28, 28, 3])
text = []

num_classes = len(np.unique(np.argmax(y_train,axis=1)))
label_to_class_name = [str(i) for i in range(num_classes)]

for i, (x, y) in enumerate(test_images):
    # Add batch axis.
    x = x[None, :, :, :]
    
    # Predict final activations, probabilites, and label.
    presm = model_wo_softmax.predict_on_batch(x)[0]
    prob = model.predict_on_batch(x)[0]
    y_hat = prob.argmax()
    
    # Save prediction info:
    text.append(("%s" % label_to_class_name[y],    # ground truth label
                 "%.2f" % presm.max(),             # pre-softmax logits
                 "%.2f" % prob.max(),              # probabilistic softmax output  
                 "%s" % label_to_class_name[y_hat] # predicted label
                ))

    for aidx, analyzer in enumerate(analyzers):
        # Analyze.
        a = analyzer.analyze(x)
        
        # Apply common postprocessing, e.g., re-ordering the channels for plotting.
        a = mnistutils.postprocess(a)
        # Apply analysis postprocessing, e.g., creating a heatmap.
        a = methods[aidx][2](a)
        # Store the analysis.
        analysis[i, aidx] = a[0]
        
        
        
# In[]

eutils = imp.load_source("utils", "../../innvestigate/examples/utils.py")

# Prepare the grid as rectengular list
grid = [[analysis[i, j] for j in range(analysis.shape[1])]
        for i in range(analysis.shape[0])]
# Prepare the labels
label, presm, prob, pred = zip(*text)
row_labels_left = [('label: {}'.format(label[i]), 'pred: {}'.format(pred[i])) for i in range(len(label))]
row_labels_right = [('logit: {}'.format(presm[i]), 'prob: {}'.format(prob[i])) for i in range(len(label))]
col_labels = [''.join(method[3]) for method in methods]

# Plot the analysis.
eutils.plot_image_grid(grid, row_labels_left, row_labels_right, col_labels,
                       file_name=os.environ.get("PLOTFILENAME", None),
                       figsize=(9,20))


# In[]
fname = 'figures/may2020/xai/'+settings['dset']+'_'+settings['test_layer']+'_'+str(
        settings['adaptive_kernel_size'])+'_Epc'+str(
                settings['epochs'])

show_shap=True
if show_shap:
    import shap
    
    # select a set of background examples to take an expectation over
    background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
    
    # explain predictions of the model on four images
    e = shap.DeepExplainer(model, background)
    # ...or pass tensors directly
    # mnist simple samps = [0,1,2,3,6]
    samps = selected_examples
    #samps = [3,5,6,9,12,13,15,23]
    # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
    shap_values = e.shap_values(x_test[samps])
    
    class_names = label_to_class_name
    # plot the feature attributions
    plt.figure()
    shap.image_plot(shap_values, (x_test[samps]+2)*64, 
                    np.tile(np.array(class_names),(len(samps),1)),sharetitles=True)


    shap_values,indexes = e.shap_values(x_test[samps], 
                                        ranked_outputs=2)
    index_names = np.vectorize(lambda x: class_names[x])(indexes)

    shap.image_plot(shap_values, (x_test[samps]+2)*64, index_names)


# concat both
    def repmat_norm(imx,a):
        imx_pos = imx*(imx>=0).astype('int')
        mn = np.min(imx_pos)
        mx = np.max(imx_pos)
        imx_pos = (imx_pos-mn)/(mx-mn)
        #imx_pos[imx_pos==0]=0.5
        
        imx_neg = np.abs(imx*(imx<0).astype('int'))
        mn = np.min(imx_neg)
        mx = np.max(imx_neg)
        imx_neg = (imx_neg-mn)/np.abs(mx-mn)
        #imx_neg[imx_neg==0]=0.5
        b = 0.9+0.05*(a-np.min(a))/(np.max(a)-np.min(a))
        imx3 = b
        imx3 = np.repeat(imx3,3,axis=2)
        #imx3[:,:,0] = 1.0
        imx3[:,:,0] += imx_pos[:,:,0]
        imx3[:,:,1] -= imx_pos[:,:,0]
        imx3[:,:,2] -= imx_pos[:,:,0]
        #imx3[:,:,1] = 0.1
        #imx3[:,:,2] = 1.0
        imx3[:,:,2] += imx_neg[:,:,0]
        imx3[:,:,1] -= imx_neg[:,:,0]
        imx3[:,:,0] -= imx_neg[:,:,0]
        
        print(np.max(imx3),np.min(imx3))
        
        #imx3 = (imx3-np.min(imx3))/(np.max(imx3)-np.min(imx3))
        
        
        return imx3
    
    for i,m in enumerate(grid):
        
        grid[i].append(repmat_norm(shap_values[0][i],x_test[samps[i]]))
        grid[i].append(repmat_norm(shap_values[1][i],x_test[samps[i]]))
            
    col_labels.append('Shap 1st')
    col_labels.append('Shap 2nd')
    
    eutils.plot_image_grid(grid, row_labels_left, row_labels_right, col_labels,
                       file_name=os.environ.get("PLOTFILENAME", fname+".png"),
                       figsize=(9,20))