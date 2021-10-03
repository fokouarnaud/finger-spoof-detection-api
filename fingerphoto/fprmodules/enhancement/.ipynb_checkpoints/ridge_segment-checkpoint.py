# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 23:04:30 2016

@author: utkarsh
"""



# RIDGESEGMENT - Normalises fingerprint image and segments ridge region
#
# Function identifies ridge regions of a fingerprint image and returns a
# mask identifying this region.  It also normalises the intesity values of
# the image so that the ridge regions have zero mean, unit standard
# deviation.
#
# This function breaks the image up into blocks of size blksze x blksze and
# evaluates the standard deviation in each region.  If the standard
# deviation is above the threshold it is deemed part of the fingerprint.
# Note that the image is normalised to have zero mean, unit standard
# deviation prior to performing this process so that the threshold you
# specify is relative to a unit standard deviation.
#
# Usage:   [normim, mask, maskind] = ridgesegment(im, blksze, thresh)
#
# Arguments:   im     - Fingerprint image to be segmented.
#              blksze - Block size over which the the standard
#                       deviation is determined (try a value of 16).
#              thresh - Threshold of standard deviation to decide if a
#                       block is a ridge region (Try a value 0.1 - 0.2)
#
# Returns:     normim - Image where the ridge regions are renormalised to
#                       have zero mean, unit standard deviation.
#              mask   - Mask indicating ridge-like regions of the image, 
#                       0 for non ridge regions, 1 for ridge regions.
#              maskind - Vector of indices of locations within the mask. 
#
# Suggested values for a 500dpi fingerprint image:
#
#   [normim, mask, maskind] = ridgesegment(im, 16, 0.1)
#
# See also: RIDGEORIENT, RIDGEFREQ, RIDGEFILTER

### REFERENCES

# Peter Kovesi         
# School of Computer Science & Software Engineering
# The University of Western Australia
# pk at csse uwa edu au
# http://www.csse.uwa.edu.au/~pk


import numpy as np
import cv2

def normalise(img,mean,std):
    normed = (img - np.mean(img))/(np.std(img));    
    return(normed)
    
def ridge_segment_old1(im,blksze,thresh):
    
    rows,cols = im.shape;    
    
    im = cv2.equalizeHist(im)
    #im = normalise(im, 0, 1);  # normalise to get zero mean and unit standard deviation

    new_rows =  np.int(blksze * np.ceil((np.float(rows))/(np.float(blksze))))
    new_cols =  np.int(blksze * np.ceil((np.float(cols))/(np.float(blksze))))
    
    padded_img = np.zeros((new_rows,new_cols));
    stddevim = np.zeros((new_rows,new_cols));
    
    padded_img[0:rows][:,0:cols] = im;
    
    for i in range(0,new_rows,blksze):
        for j in range(0,new_cols,blksze):
            block = padded_img[i:i+blksze][:,j:j+blksze];
            stddevim[i:i+blksze][:,j:j+blksze] = np.std(block)*np.ones(block.shape)
    
    stddevim = stddevim[0:rows][:,0:cols]
                    
    mask = stddevim > thresh;
    
    mean_val = np.mean(im[mask]);
    
    std_val = np.std(im[mask]);
    
    normim = (im - mean_val)/(std_val);
    
    return(normim,mask)


def ridge_segment_old2(im,blksze,thresh):
    
    rows,cols = im.shape
    
    im = cv2.equalizeHist(im)
    #im = normalise(im, 0, 1);  # normalise to get zero mean and unit standard deviation

    new_rows =  np.int(blksze * np.ceil((np.float(rows))/(np.float(blksze))))
    new_cols =  np.int(blksze * np.ceil((np.float(cols))/(np.float(blksze))))
    
    padded_img = np.zeros((new_rows,new_cols));
    stddevim = np.zeros((new_rows,new_cols));
    mask =np.zeros((new_rows,new_cols),np.uint8)
    padded_img[0:rows][:,0:cols] = im;

    
    for i in range(0,new_rows,blksze):
        for j in range(0,new_cols,blksze):
            block = padded_img[i:i+blksze][:,j:j+blksze];
            mean, std = cv2.meanStdDev(block)
            std_val = std[0][0]
            stddevim[i:i+blksze][:,j:j+blksze] = cv2.multiply(np.ones(block.shape),std_val)
            mask[i:i+blksze][:,j:j+blksze] =  cv2.multiply(np.ones(block.shape), 1 if std_val > thresh else 0)
    
    stddevim = stddevim[0:rows][:,0:cols]
                    
    mask = mask[0:rows][:,0:cols]
    
   # mean_val = np.mean(im[mask])
    
    #std_val = np.std(im[mask])
    
    #normim = (im - mean_val)/(std_val)
    
    mean, std = cv2.meanStdDev(im, mask=mask)
    result=cv2.subtract(im,mean[0][0])
    mean, std = cv2.meanStdDev(result, mask=mask)

    result = cv2.divide(result,std[0][0])
    
    return(result,mask)



def ridge_segment(source,blockSize,threshold):
    
    
    source = cv2.equalizeHist(source)
    source = source.astype(np.float32)
    
    result =np.zeros(source.shape, np.float32)
    mask =np.zeros(source.shape, np.uint8)
    
    height, width = source.shape
    
    #source=normalise(source,0,1)

    widthSteps =width// blockSize
    heightSteps = height// blockSize

    scalarBlack = (0, 0, 0)
    scalarWhite =(255, 255, 255)


    windowMask  = np.zeros(source.shape, np.uint8)

    for y in range(1,heightSteps+1) :
        for  x in range(1,widthSteps+1) :
            roi_x,roi_y,roi_width, roi_height= ((blockSize) * (x - 1), (blockSize) * (y - 1), blockSize, blockSize)
            windowMask[:] = 0
            windowMask = cv2.rectangle(windowMask, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), scalarWhite, -1, 8, 0)
            y1=roi_y
            y2=y1+roi_height
            x1=roi_x
            x2=x1+roi_width

            window = source[y1:y2][:, x1:x2]
            # display first load image
            mean, std = cv2.meanStdDev(window)
            mean_val = mean[0][0]
            std_val = std[0][0]
            
            result[windowMask>0] = std_val
            
            mask[windowMask>0] = 1 if (std_val >=threshold) else 0


    # get mean and standard deviation
    mean, std = cv2.meanStdDev(source, mask=mask)
    result=cv2.subtract(source,mean[0][0])
    mean, std = cv2.meanStdDev(result, mask=mask)

    result = cv2.divide(result,std[0][0])

    

    return (result,mask)

    