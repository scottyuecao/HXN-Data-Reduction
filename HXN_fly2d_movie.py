#==============================================================================
# ver 1. Yue Cao (ycao@colorado.edu)
#
# Last modified: 2018-01-05
#
# Creating a fly_2d movie from real data
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

sid = 39451
fluoKey = 'Ta_L'

xbin = 10
ybin = 10

xRange = [-12.5, 12.5]
yRange = [-12.5, 12.5]
xSize = 100
ySize = 100

fluomin = 0
fluomax = 650

summin = 0
summax = 14000

imgmin = 0.
imgmax = 100.

outNm = 'out_'
outFd = os.path.join(os.getcwd(), 'tiff2')

dpi = 400

#==============================================================================
# Main program
#==============================================================================

if os.path.isdir(outFd):
    shutil.rmtree(outFd)
    os.makedirs(outFd)
else:
    os.makedirs(outFd)

h5nm = os.path.join(os.getcwd(), 'scan_'+str(sid)+'.h5')
h5 = h5py.File(h5nm, mode='r')

h5rnm = os.path.join(os.getcwd(), 'Scan'+str(sid)+'_r.h5')
h5r = h5py.File(h5rnm, mode='r')

roisum = h5r['roi_sum']
fluo = h5r[fluoKey]

imgStack = h5['entry/instrument/detector/data']

tempfluo = np.ones((ySize, xSize))
tempsum = np.ones((ySize, xSize))

tempfluo = np.where(tempfluo>0, np.nan, 0)
tempsum = np.where(tempsum>0, np.nan, 0)

plt.figure(figsize=(9,3))

counter = 0
for j in range(int(ySize/ybin)):
    for i in range(int(xSize/xbin)):
        for m in range(ybin):
            for l in range(xbin):
                tempfluo[j*ybin+m, i*xbin+l] = fluo[j*ybin+m, i*xbin+l]
                tempsum[j*ybin+m, i*xbin+l] = roisum[j*ybin+m, i*xbin+l]
        
        tempimg = imgStack[j*ybin*ySize+i*xbin]
        
        plt.clf()
        ax1 = plt.subplot(131)
        plt.imshow(tempfluo, vmin=fluomin, vmax = fluomax,
                   extent=[xRange[0], xRange[1], yRange[1], yRange[0]])
        plt.colorbar()
        ax1.set_title('Vortex')

        ax2 = plt.subplot(132)
        plt.imshow(tempsum, vmin=summin, vmax = summax,
                   extent=[xRange[0], xRange[1], yRange[1], yRange[0]])
        plt.colorbar()
        ax2.set_title('ROI Sum')

        ax3 = plt.subplot(133)
        plt.imshow(tempimg, vmin=imgmin, vmax = imgmax)
        plt.xlim([70,130])
        plt.ylim([120, 60])
        plt.colorbar()
        ax3.set_title('Merlin CCD')

        #plt.tight_layout()
        plt.suptitle('Scan '+str(sid), fontsize=14)
        plt.subplots_adjust(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.3)

        outpath = os.path.join(outFd, 'out_'+str(counter)+'.tiff')
        plt.savefig(outpath, dpi=dpi)
        counter += 1