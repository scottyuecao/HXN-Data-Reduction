#==============================================================================
# ver 1. Yue Cao (ycao@colorado.edu)
#
# Last modified: 2018-01-05
#
# Creating a movie of roi_sum vs theta from real data
#
#==============================================================================


# importing system packages
import os
import sys
import h5py
import time
import tifffile
import itertools
import shutil

# import all plotting packages
import matplotlib.pyplot as plt
from matplotlib import colors, cm, animation
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

# Image codes
from ImgUtil import *

#==============================================================================
# User inputs
#==============================================================================

xRange = [-12.5, 12.5]
yRange = [-12.5, 12.5]
xSize = 100
ySize = 100

sum_nm = 'roi_sum.tiff'
csv_nm = 'sorted_files.csv'

outNm = 'out_'
outFd = os.path.join(os.getcwd(), 'tiff')

vmin = 0
vmax = 14000

dpi = 400

#==============================================================================
# Main program
#==============================================================================

fd = os.getcwd()

sumpath = os.path.join(fd, sum_nm)
sumstk = tifffile.imread(sumpath)

csvpath = os.path.join(fd, csv_nm)
sorted_files = pd.read_csv(csvpath, sep='\t', index_col=0)
thetaList = sorted_files['Theta']

if os.path.isdir(outFd):
    shutil.rmtree(outFd)
    os.makedirs(outFd)
else:
    os.makedirs(outFd)

fig = plt.figure(figsize=(5,4))
for i in range(sumstk.shape[0]):
    plt.clf()
    plt.imshow(sumstk[i], vmin=vmin, vmax=vmax,
               extent=[xRange[0], xRange[1], yRange[1], yRange[0]])
    plt.colorbar()
    plt.title('Theta = {:.2f} deg'.format(thetaList[i]))
    
    outpath = os.path.join(outFd, 'out_'+str(i)+'.tiff')
    plt.savefig(outpath, dpi=dpi)