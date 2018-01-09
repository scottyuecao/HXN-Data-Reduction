#==============================================================================
# ver 1. Yue Cao (ycao@colorado.edu)
#
# Last modified: 2017-12-13
#
# This program compiles selected keys from the reduced h5 into a stack of
# images.
#==============================================================================


# importing system packages
import os
# import sys
import h5py
# import time

# import all plotting packages
#import matplotlib.pyplot as plt
#from matplotlib import colors, cm
import tifffile

# importing the workhorse
import numpy as np
import pandas as pd

#==============================================================================
# All inputs and outputs
#==============================================================================

# A list of reduced h5 file names.
sidList = list(np.arange(39439, 39466))
h5rList = ['Scan'+str(sid)+'_r.h5' for sid in sidList]

# A list of corresponding thetas.
thetaList = list(np.linspace(-7.96, -6.92, num=27))

# Folder path
fd = os.getcwd()

# A list of needed datasets in the reduced h5s.
keyList = ['Ta_L', 'Ta_L_n', 'roi_sum', 'roi_comx', 'roi_comy']

#==============================================================================
# Functions
#==============================================================================

def save2TiffStk(imageArray, imName):
    imageArray = imageArray.astype(np.int32)
    fname = imName+'.tiff'
    tifffile.imsave(fname, imageArray)


def genImgStack(h5rList, thetaList, fd, keyList, saveTiff=False):
    # The following code do not work for the moment.
    # From Stack Overflow: There appears to be an issue with the latest version of
    # Numpy. A recent change made it an error to treat a single-element array as a 
    # scalar for the purposes of indexing.

    # flags = np.argsort(thetaList, kind='mergesort')
    # thetaList = thetaList[flags]
    # h5rList = h5rList[flags]

    flags = np.argsort(thetaList, kind='mergesort')
    sortedTheta = [thetaList[i] for i in flags]
    sortedH5r = [h5rList[i] for i in flags]

    imgStackDict = {}
    for key in keyList:
        imgStackDict[key] = []

    for counter, nm in enumerate(sortedH5r):
        fpath = os.path.join(fd, nm)
        h5 = h5py.File(fpath, 'r')
        for key in keyList:
            if counter:
                imgStackDict[key].append(np.array(h5[key]))
            else:
                imgStackDict[key] = [np.array(h5[key])]
    
    for key in keyList:
        imgStackDict[key] = np.array(imgStackDict[key])
        if saveTiff:
            save2TiffStk(imgStackDict[key], str(key))
    
    if saveTiff:
        data = np.vstack((sortedTheta, sortedH5r)).transpose()
        sorted_files = pd.DataFrame(data=data, columns=('Theta', 'Reduced h5 file'))
        sorted_files.to_csv('sorted_files.csv', sep='\t')
    
    return sorted_files


#==============================================================================
# Functions
#==============================================================================

genImgStack(h5rList, thetaList, fd, keyList, saveTiff=True)