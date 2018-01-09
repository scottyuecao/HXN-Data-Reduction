#==============================================================================
# ver 1. Yue Cao (ycao@colorado.edu)
#
# Last modified: 2017-12-13
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
import tifffile

# ndimage.measurements.center_of_mass could not resolve nan's.
# Alternatives:
# 1) Using 0 in place of nan's. This will generate the same center of mass.
# 2) Using masked array. Number 1 in the mask invalidate the entry in the array.
from scipy import ndimage, signal

#==============================================================================
# Utility functions for aligning images at different theta angles.
#==============================================================================

def flat2D(in2d, roi=None):
    '''
    Input:
    in2d:   A 2d image, of the shape (ySize, xSize).
    roi:    A list. [xMin, xMax, yMin, yMax].
    '''
    if roi==None:
        bg = in2d.mean()
    else:
        bg = in2d[roi[2]:roi[3], roi[0]:roi[1]].mean()
    return bg


def corr2d(im1, im2, roi=None, plot=False):
    '''
    Input:
    im1, im2:   
    Output:
    Note axis=0 is the y dimension.
    '''
    bg = flat2D(im1, roi=roi)
    temp1 = np.copy(im1) - bg
    temp2 = np.copy(im2) - bg
    corr = signal.correlate2d(temp1, temp2)

    corrMax = corr.max()
    (corrIDy, corrIDx) = np.unravel_index(corr.argmax(), corr.shape)
    corrIDy = corrIDy-im1.shape[0]+1
    corrIDx = corrIDx-im1.shape[1]+1

    if plot:
        plt.figure()
        plt.clf()
        plt.imshow(corr, extent=[-im1.shape[0]+1, im1.shape[0]-1, im1.shape[1]-1, -im1.shape[1]+1])
    
    return corrIDx, corrIDy


def alignImgs(im1, im2, roi=None):
    '''
    This function aligns and crops 2 images to the same size.

    Input:
    im1:        The reference image.
    im2:        The image that will be aligned to im1.
    '''
    corrIDx, corrIDy = corr2d(im1, im2, roi=roi)

    if corrIDy>0:
        if corrIDx>0:
            out1 = im1[corrIDy:, corrIDx:].copy()
            out2 = im2[:-corrIDy, :-corrIDx].copy()
        else:
            out1 = im1[corrIDy:, :corrIDx].copy()
            out2 = im2[:-corrIDy, -corrIDx:].copy()
    else:
        if corrIDx>0:
            out1 = im1[:corrIDy, corrIDx:].copy()
            out2 = im2[-corrIDy:, :-corrIDx].copy()
        else:
            out1 = im1[:corrIDy, :corrIDx].copy()
            out2 = im2[-corrIDy:, -corrIDx:].copy()
    
    return out1, out2


def corrImgs(ref_Stk, ref_num, fd=None, roi=None):
    '''
    This function

    Input:

    ref_Stk:        The reference image stack.
    ref_num:        The slice number in the ref_Stk which all other
                    slices align to.
    fd:             The file dir. Default is current folder.
    roi:            The roi for calculating the background.
                    Default is the entire map.
    '''

    if fd==None:
        fd = os.getcwd()

    # Reading the ref tiff stack
    temppath = os.path.join(fd, ref_Stk)
    maps = tifffile.imread(temppath)

    # Getting the ref map.
    im0 = maps[ref_num]

    # Getting the relative shifts.
    corrIDxList = []
    corrIDyList = []
    for i in range(maps.shape[0]):
        im1 = maps[i]
        corrIDx, corrIDy = corr2d(im0, im1, roi=roi)
        corrIDxList.append(corrIDx)
        corrIDyList.append(corrIDy)

    return corrIDxList, corrIDyList