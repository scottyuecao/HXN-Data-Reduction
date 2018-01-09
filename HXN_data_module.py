#==============================================================================
# ver 4. Yue Cao (ycao@colorado.edu)
#
# Last modified: 2017-12-13
#
# The codes are designed for best efficiency and RAM. We will
#
# 1) Load h5's without converting to numpy.array.
#
# 2) Do not return an image stack at the end of a function unless needed.
#
# 3) Running one for-loop to extract roi and comx/comy at the same time.
#
#
# If we load the h5 as a numpy.array, it is 'uint32' type. 10,000 images of
# (330, 263) use around 3.5 GB. Using 'uint8' will save a factor of 4 RAM.
# Using float will increase RAM by a factor of 2.
#
#
# To simplify the code, we will not check for array size mismatches and will let
# the generic numpy to take care of it.
#
#
# Coordinate systems used in the code:
#
# 1) All the real-space images are shown as if we are looking at the sample
# headon. Note 'zpssy' is inversed, +zpssy points down. +zpssx points rightward.
#
# 2) When reshape a numpy.array, the size input should be (ySize, xSize).
#==============================================================================


# importing system packages
import os
import sys
import h5py
import time

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


def loadVortex(sid, folder='', elements=['Si_K'], plot=False):
    '''
    This function obtains all the scalars from the vortex folder

    Inputs:
    sid:        Scan id. Can be either an int or a str.
    folder:     The parent dir of the vortex folder. Default is the current file folder.
    elements:   A list of chemical elements whose fluorescence data will be collected.
                Both the elements and the edge need to be included.
    plot:       Whether to plot the fluo map. Default is False.
    '''

    if folder=='':
        folder = os.getcwd()
    vtxFolder = os.path.join(folder, 'output_txt_scan2D_'+str(sid))

    # vor is a dict containing the fluo maps of all elements from vortex fitting.
    vor = {}
    for element in elements:
        tempfn = 'detsum_'+element+'_'+str(sid)+'.txt'
        temppath = os.path.join(vtxFolder, tempfn)
        vor[element] = np.genfromtxt(temppath)

        tempfn = 'detsum_'+element+'_'+str(sid)+'_norm.txt'
        temppath = os.path.join(vtxFolder, tempfn)
        vor[element+'_n'] = np.genfromtxt(temppath)

    tempfn = 'x_pos_'+str(sid)+'.txt'
    temppath = os.path.join(vtxFolder, tempfn)
    vor['zpssx'] = np.genfromtxt(temppath)
        
    tempfn = 'y_pos_'+str(sid)+'.txt'
    temppath = os.path.join(vtxFolder, tempfn)
    vor['zpssy'] = np.genfromtxt(temppath)

    # Actual scan range of zpssx and zpssy, in units of microns.
    # The nominal scan range could be slightly different.
    xMin = vor['zpssx'].min()
    xMax = vor['zpssx'].max()
    yMin = vor['zpssy'].min()
    yMax = vor['zpssy'].max()

    if plot:
        plt.figure()
        plt.clf()
        
        # Note that (4+1)/2 = 2, and (5+1)/2=3.
        numCol = 2
        numRow = (len(elements)+1)/2
        for num, element in enumerate(elements):
            ax = plt.subplot(numCol,numRow,num+1)
            plt.imshow(vor[element], cmap=cm.magma, extent=[xMin, xMax, yMax, yMin])
            plt.colorbar()
            ax.set_title(str(element))

        plt.tight_layout()
        plt.suptitle('Scan '+str(sid)+' Fluorescence', fontsize=16)
        plt.subplots_adjust(top=0.88)
        plt.show()

    return vor


def loadSclr(sid, xRange, yRange, xSize, ySize, folder='', elements=['Si'], plot=False):
    '''
    This function obtains all the scalar values from the txt file at HXN.
    
    Inputs:
    
    sid:        Scan id. Can be either an int or a str.
    xRange:     Scan range of zpssx, in units of microns. Input is a list.
    yRange:     Scan range of zpssy, in units of microns. Input is a list.
    xSize:      Total steps along zpssx.
    ySize:      Total steps along zpssx.
    folder:     The dir of the txt file. Default is the current file folder.
    elements:   A list of chemical elements whose fluorescence data will be collected.
    plot:       Whether to plot the fluo map. Default is False.
    
    Outputs:
    
    fluo:       A dict of fluo maps. Each is a numpy.array.
    pos:        A pd.DataFrame of motor positions. pos.index starts from 0.
                Thus pos.loc[] and pos.iloc[] will return the same value.
    '''

    if folder=='':
        folder = os.getcwd()
    txtnm = os.path.join(folder, 'scan_'+str(sid)+'.txt')
    
    # pandas provides a better way for dealing with the header than np.genfromtxt for HXN.
    sclr = pd.read_table(txtnm, index_col=0)
    
    # fluo is a dict containing the fluo maps of all elements.
    fluo = {}
    for element in elements:
        fluo[element] = np.zeros(xSize*ySize)
        for col in sclr.columns:
            if element==col[-len(element):]:
                fluo[element] += sclr[col]
        fluo[element] = np.reshape(fluo[element], (ySize, xSize))
    
    # pos is a pd.DataFrame
    pos = {}
    pos['zpssx'] = sclr['zpssx']
    pos['zpssy'] = sclr['zpssy']
    tempx, tempy = np.meshgrid(np.linspace(xRange[0], xRange[1], num=xSize), np.linspace(yRange[0], yRange[1], num=ySize))
    pos['calcx'] = tempx.flatten()
    pos['calcy'] = tempy.flatten()
    pos = pd.DataFrame(data=pos)
    # Note pos.index starts from an index of 1. Change pos.index so that it starts from 0.
    pos.index = pos.index-1

    # Nominal scan range of zpssx and zpssy, in units of microns.
    xMin = xRange[0]
    xMax = xRange[1]
    yMin = yRange[0]
    yMax = yRange[1]
    
    if plot:
        plt.figure()
        plt.clf()
        
        # Note that (4+1)/2 = 2, and (5+1)/2=3.
        numCol = 2
        numRow = (len(elements)+1)/2
        for num, element in enumerate(elements):
            ax = plt.subplot(numCol,numRow,num+1)
            plt.imshow(fluo[element], cmap=cm.magma, extent=[xMin, xMax, yMax, yMin])
            plt.colorbar()
            ax.set_title(str(element))

        plt.tight_layout()
        plt.suptitle('Scan '+str(sid)+' Fluorescence', fontsize=16)
        plt.subplots_adjust(top=0.88)
        plt.show()

        # This is to check the accuracy of the zone plate motor positions ('zpssx' and 'zpssy').
        diffx = pos['zpssx']-pos['calcx']
        diffy = pos['zpssy']-pos['calcy']
        
        plt.figure()
        plt.clf()
        
        ax1 = plt.subplot(221)
        plt.imshow(diffx.values.reshape((ySize, xSize)), cmap=cm.magma)
        plt.colorbar()
        ax1.set_title('zpssx')
        
        ax2 = plt.subplot(222)
        sns.distplot(diffx)
        ax2.set_title('zpssx')
        
        ax3 = plt.subplot(223)
        plt.imshow(diffy.values.reshape((ySize, xSize)), cmap=cm.magma)
        plt.colorbar()
        ax3.set_title('zpssy')
        
        ax4 = plt.subplot(224)
        sns.distplot(diffy)
        ax4.set_title('zpssy')
        
        plt.tight_layout()
        plt.suptitle('Scan '+str(sid)+' Readback-Setpoint', fontsize=16)
        plt.subplots_adjust(top=0.88)
        plt.show()
    
    return fluo, pos


def reduceH5(h5id, xRange, yRange, xSize, ySize, folder='', msk=[], fluo_thre=[], plot=False):
    '''
    This function extracts all features of a given det image, and maps them in real space.
    
    Inputs:
    
    h5id:       Scan id. Can be either an int or a str.
    xRange:     Scan range of zpssx, in units of microns. Input is a list.
    yRange:     Scan range of zpssy, in units of microns. Input is a list.
    xSize:      Total steps along zpssx.
    ySize:      Total steps along zpssx.
    folder:     The dir of the HDF5 file. Default is the current file folder.
    msk:        A digital mask to remove powder lines from the HXN detector. A numpy.array.
                Default is an empty list.
    fluo_thre:  Fluo threshold. This is a list consisting of [fluo, element, threshold].
                'fluo' is the fluo dict generated by loadVortex or loadSclr, element is a str.
                threshold is an int.
    plot:       Whether to plot the fluo map. Default is False.
    
    Outputs:
    
    h5reduce:   A dict of feature maps. Each is a numpy.array.
    '''
    
    if folder=='':
        folder = os.getcwd()
    h5nm = os.path.join(folder, 'scan_'+str(h5id)+'.h5')
    h5 = h5py.File(h5nm, mode='r')
    
    imgStack = h5['entry/instrument/detector/data']
    
    roi_sum = np.zeros((imgStack.shape[0]))
    roi_comx = np.zeros((imgStack.shape[0]))
    roi_comy = np.zeros((imgStack.shape[0]))
    
    for counter in range(imgStack.shape[0]):
        if len(msk)!=0:
            # In the presence of a mask.
            # Using a mask in np.ma is identical to replace all the unwanted features with zero.
            # Mask generated from h5mskHXN is either 0 or 1, with 1 the unwanted regions.
            # For a 2d array, len(msk)=msk.shape[0]
            img = imgStack[counter]*(1-msk)
        else:
            img = imgStack[counter]
            
        roi_sum[counter] = np.nansum(img)
        (roi_comx[counter], roi_comy[counter]) = ndimage.measurements.center_of_mass(img)
    
    roi_sum = np.reshape(roi_sum, (ySize, xSize))
    roi_comx = np.reshape(roi_comx, (ySize, xSize))
    roi_comy = np.reshape(roi_comy, (ySize, xSize))
    
    if len(fluo_thre)==3:
        # We have a fluo threshold in place.
        fluo = fluo_thre[0]
        element = fluo_thre[1]
        roi_sum = np.where(fluo[element]>fluo_thre[2], roi_sum, np.nan)
        roi_comx = np.where(fluo[element]>fluo_thre[2], roi_comx, np.nan)
        roi_comy = np.where(fluo[element]>fluo_thre[2], roi_comy, np.nan)
    
    h5reduce = {}
    h5reduce['roi_sum'] = roi_sum
    h5reduce['roi_comx'] = roi_comx
    h5reduce['roi_comy'] = roi_comy
    
    if plot:
        plt.figure(figsize=(6,15))
        plt.clf()

        ax1 = plt.subplot(311)
        plt.imshow(h5reduce['roi_sum'], cmap=cm.magma, extent=[xRange[0], xRange[1], yRange[1], yRange[0]])
        plt.colorbar()
        ax1.set_title('Integrated detector intensity')

        ax2 = plt.subplot(312)
        plt.imshow(h5reduce['roi_comx'], cmap=cm.magma, extent=[xRange[0], xRange[1], yRange[1], yRange[0]])
        plt.colorbar()
        ax2.set_title('Center of mass along x')

        ax3 = plt.subplot(313)
        plt.imshow(h5reduce['roi_comy'], cmap=cm.magma, extent=[xRange[0], xRange[1], yRange[1], yRange[0]])
        plt.colorbar()
        ax3.set_title('Center of mass along y')

        plt.tight_layout()
        plt.suptitle('Scan '+str(h5id), fontsize=16)
        plt.subplots_adjust(top=0.92)
        plt.show()
    
    return h5reduce


def sumH5(h5id, Nth=0, folder='', plot=False):
    '''
    This function returns the summed detector image for a given range.
    
    Inputs:
    
    h5id:       Scan id. Can be either an int or a str.
    Nth:        Summing the first N-th image. Default is zero, where all images are included.
    folder:     The dir of the HDF5 file. Default is the current file folder.
    plot:       Whether to plot the summed det image. Default is False.
    
    Outputs:
    
    h5sum:      A numpy.array. Sum of the n images in the sumRange.
    '''
    
    if type(h5id)==int:
        h5id = str(h5id)
    if folder=='':
        folder = os.getcwd()
    h5nm = os.path.join(folder, 'scan_'+h5id+'.h5')
    h5 = h5py.File(h5nm, mode='r')
    
    imgStack = h5['entry/instrument/detector/data']
    if Nth>0:
        # Note only the 0th to the (N-1)-th slices are included.
        h5sum = np.nansum(imgStack[:Nth], axis=0)
        print('********** Summing the first '+str(Nth)+' det image(s). **********')
    else:
        h5sum = np.nansum(imgStack, axis=0)
        print('********** Summing all det image(s). **********')
    
    if plot:
        vmin = np.percentile(h5sum, 5)
        vmax = np.percentile(h5sum, 95)

        plt.figure()
        plt.clf()
        plt.subplot(121)
        plt.imshow(h5sum, cmap=cm.magma)
        plt.colorbar()

        plt.subplot(122)
        plt.imshow(h5sum, cmap=cm.magma, vmin=vmin, vmax=vmax)
        plt.colorbar()
    
    return h5sum


#==============================================================================
# Utility functions for getting a specific frame.
#==============================================================================


def getFrameID(pos, xPos, yPos):
    '''
    This function returns the frame number of a given (xPos, yPos).
    
    Inputs:
    
    pos:        A pd.DataFrame of motor positions.
    xPos:       Float. The x position on the sample.
    yPos:       Float. The y position on the sample.
    
    Output:
    
    frameID:    Int. The index of the frame (from 0 to xSize*ySize-1).
    '''
    
    # distance is a pd.Series
    distance = (pos['zpssx']-xPos)**2+(pos['zpssy']-yPos)**2
    frameID = distance.idxmin()
    
    return frameID


def getDetFrame(h5id, frameID, folder='', plot=True):
    '''
    This function returns the detector image for a given frame ID.
    
    Inputs:
    
    h5id:       Scan id. Can be either an int or a str.
    frameID:    Int. The index of the frame (from 0 to xSize*ySize-1).
    folder:     The dir of the HDF5 file. Default is the current file folder.
    plot:       Whether to plot the det image. Default is True.
    
    Outputs:
    
    img:        The det image. A numpy.array.
    '''

    if type(h5id)==int:
        h5id = str(h5id)
    if folder=='':
        folder = os.getcwd()
    h5nm = os.path.join(folder, 'scan_'+h5id+'.h5')
    h5 = h5py.File(h5nm, mode='r')
    
    img = np.array(h5['entry/instrument/detector/data'][frameID])
    
    if plot:
        plt.figure()
        plt.clf()
        plt.imshow(img, cmap=cm.magma)
        plt.colorbar()
    
    return img

#==============================================================================
# Utility function for generating mask.
#==============================================================================


def mskH5(h5sum, center=[], radius=[50, 50], lb=10., plot=False):
    '''
    This function returns a mask. Pixels of the mask with a non-zero value will be masked.
    In our case, the mask value is either 0 or 1.
    
    Inputs:
    
    h5sum:      The summed h5 image stacks.
    center:     The center of mass of the main feature.
    radius:     The radius around the center of mass.
    lb:         Lower bound. Any value above the lb but not in the main feature will be masked.
    plot:       Whether to plot. Default is False.
    
    Outputs:
    
    h5msk:      A numpy.array.
    '''
    
    # Keeping the main feature.
    if center==[]:
        center = list(ndimage.measurements.center_of_mass(h5sum))
    tempx, tempy = np.meshgrid(np.arange(h5sum.shape[1]), np.arange(h5sum.shape[0]))
    distance = ((tempx-center[1])**2/radius[0]**2+(tempy-center[0])**2/radius[1]**2)**0.5
    h5msk = np.where(distance>1., 1, 0)

    # Marking the unwanted features for the mask.
    temp = h5sum*h5msk
    h5msk = np.where(temp>lb, 1, 0)
    
    if plot:
        vmin = np.percentile(h5sum, 5)
        vmax = np.percentile(h5sum, 95)
    
        plt.figure()
        plt.clf()
        plt.subplot(131)
        plt.imshow(h5sum, cmap=cm.magma, vmin=vmin, vmax=vmax)
        plt.subplot(132)
        plt.imshow(h5msk, cmap=cm.magma)
        plt.subplot(133)
        plt.imshow(h5sum*(1-h5msk), cmap=cm.magma, vmin=vmin, vmax=vmax)
        plt.tight_layout()
        plt.show()
    
    return h5msk

#==============================================================================
# Utility function for stacking reduced datasets together.
#==============================================================================

def concatenateH5(h5reduces, plot=True):
    '''
    This function concatenates several h5reduce dict's.
    
    Inputs:
    
    h5reduces:  A tuple of h5reduce dicts.
    plot:       Whether to plot. Default is True.
    
    Output:
    
    outreduce:  A h5reduce file. A dict.
    '''
    
    roi_sum = h5reduces[0]['roi_sum']
    roi_comx = h5reduces[0]['roi_comx']
    roi_comy = h5reduces[0]['roi_comy']
    
    for h in h5reduces[1:]:
        roi_sum = np.concatenate((roi_sum, h['roi_sum']), axis=0)
        roi_comx = np.concatenate((roi_comx, h['roi_comx']), axis=0)
        roi_comy = np.concatenate((roi_comy, h['roi_comy']), axis=0)
    
    outreduce = {}
    outreduce['roi_sum'] = roi_sum
    outreduce['roi_comx'] = roi_comx
    outreduce['roi_comy'] = roi_comy
    
    if plot:
        plt.figure(figsize=(6,15))
        plt.clf()
    
        ax1 = plt.subplot(311)
        plt.imshow(outreduce['roi_sum'], cmap=cm.magma)
        plt.colorbar()
        ax1.set_title('Integrated detector intensity')

        ax2 = plt.subplot(312)
        plt.imshow(outreduce['roi_comx'], cmap=cm.magma)
        plt.colorbar()
        ax2.set_title('Center of mass along x')

        ax3 = plt.subplot(313)
        plt.imshow(outreduce['roi_comy'], cmap=cm.magma)
        plt.colorbar()
        ax3.set_title('Center of mass along y')

        plt.tight_layout()
        plt.suptitle('Concatenated output', fontsize=16)
        plt.subplots_adjust(top=0.92)
        plt.show()
    
    return outreduce
