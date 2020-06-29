#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 21:13:12 2018

@author: boray
"""

# plot utilities
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
def paper_fig_settings(addtosize=0):
    #mpl.style.use('seaborn-white')
    mpl.rc('figure',dpi=144)
    mpl.rc('text', usetex=False)
    mpl.rc('axes',titlesize=16+addtosize)
    mpl.rc('xtick', labelsize=12+addtosize)
    mpl.rc('ytick', labelsize=12+addtosize)
    mpl.rc('axes', labelsize=14+addtosize)
    mpl.rc('legend', fontsize=10+addtosize)
    mpl.rc('image',interpolation=None)
    #plt.rc('text.latex', preamble=r'\usepackage{cmbright}')

from datetime import datetime
def save_fig(fig_id, tight_layout=True, prefix="", postfix=""):
    now = datetime.now()
    timestr = now.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(prefix, "outputs", 
                        postfix, fig_id + "_"+timestr+".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)