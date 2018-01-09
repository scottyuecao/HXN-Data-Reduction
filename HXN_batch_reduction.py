#==============================================================================
# ver 1. Yue Cao (ycao@colorado.edu)
#
# Last modified: 2017-12-13
#
# Processing the vortex, txt and h5 for the same scan.
# 
# For a given Bragg peak.
# 
#==============================================================================
# importing system packages
import os
import sys
import h5py
import time

# import all plotting packages
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import seaborn as sns

# importing the workhorse
import numpy as np
import pandas as pd

# ndimage.measurements.center_of_mass could not resolve nan's.
# Alternatives:
# 1) Using 0 in place of nan's. This will generate the same center of mass.
# 2) Using masked array. Number 1 in the mask invalidate the entry in the array.
from scipy import ndimage, signal

# HXN codes
from HXN_data_module import *

#==============================================================================
# All inputs and outputs
#==============================================================================

sids = np.arange(39439, 39466)
xRange = [-12.5, 12.5]
yRange = [-12.5, 12.5]
xSize = 100
ySize = 100

vorf = os.getcwd()
vorel = ['Si_K', 'Ta_L']
vorplot = False


txtf = os.getcwd()
txtel = ['Si', 'Ta']
txtplot = False

h5f = os.getcwd()
msk=[]
fluo_thre=[]
h5plot = False


for sid in sids:
    try:
        vor = loadVortex(sid, folder=vorf, elements=vorel, plot=vorplot)
        fluo, pos = loadSclr(sid, xRange, yRange, xSize, ySize, folder=txtf, elements=txtel, plot=txtplot)
        reduced = reduceH5(sid, xRange, yRange, xSize, ySize, folder=h5f, msk=msk, fluo_thre=fluo_thre, plot=h5plot)
        out = h5py.File('Scan'+str(sid)+'_r.h5')
        ## Defining what to write in h5
        for el in vorel:
            out.create_dataset(el,data=vor[el])
            out.create_dataset(el+'_n',data=vor[el+'_n'])
        
        out.create_dataset('zppos',data=pos)

        out.create_dataset('roi_sum',data=reduced['roi_sum'])
        out.create_dataset('roi_comx',data=reduced['roi_comx'])
        out.create_dataset('roi_comy',data=reduced['roi_comy'])
        print('********** Scan %5d Reduction Done. **********' %sid)
    except IOError:
        print('********** Scan %5d Reduction Failed. **********' %sid)
    