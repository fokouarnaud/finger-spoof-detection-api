import cv2
import numpy as np
import math 
import copy
from skimage.measure import label  



def get_skin_mask_cmyk(bgr):
   
    bgr = bgr.astype(float)/255.
    # Extract channels
    with np.errstate(invalid='ignore', divide='ignore'):
        K = 1 - np.max(bgr, axis=2)
        C = (1-bgr[...,2] - K)/(1-K)
        M = (1-bgr[...,1] - K)/(1-K)
        Y = (1-bgr[...,0] - K)/(1-K)

    # Convert the input BGR image to CMYK colorspace
    CMYK = (np.dstack((C,M,Y,K)) * 255).astype(np.uint8)

    # Split CMYK channels
    Y, M, C, K = cv2.split(CMYK)

    np.isfinite(C).all()
    np.isfinite(M).all()
    np.isfinite(K).all()
    np.isfinite(Y).all()
    
    return M

def get_skin_mask_ycrcb(img):
    min_YCrCb = np.array([0,133,77],np.uint8)
    max_YCrCb = np.array([235,173,127],np.uint8)

    # Get pointer to video frames from primary device
    imageYCrCb = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
    return skinRegionYCrCb

def get_skin_mask_hsv(img):
    lower_1 = np.array([0,133,77],np.uint8)
    upper_1 = np.array([235,173,127],np.uint8)
    lower = np.array([0, 48, 80], np.uint8)
    upper = np.array([20, 255, 255], np.uint8)

    #converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #skin color range for hsv color space 
    HSV_mask = cv2.inRange(img_HSV, lower, upper) 
    return HSV_mask

def get_skin_mask_hsv_ycrcb(img):
    
    lower = np.array([0, 48, 80], np.uint8)
    upper = np.array([20, 255, 255], np.uint8)
    
    min_YCrCb = np.array([0,133,77],np.uint8)
    max_YCrCb = np.array([235,173,127],np.uint8)

    #converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #skin color range for hsv color space 
    HSV_mask = cv2.inRange(img_HSV, lower,upper) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #skin color range for hsv color space 
    YCrCb_mask = cv2.inRange(img_YCrCb, min_YCrCb,max_YCrCb) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #merge skin detection (YCbCr and hsv)
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


    HSV_result = cv2.bitwise_not(HSV_mask)
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    global_result=cv2.bitwise_not(global_mask)
    
    return  global_result


def rotation(img):
    img = cv2.transpose(img)
    img = cv2.flip(img,0)
    return img

def fix_orientation(image,orientation):
    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW

    if type(orientation) is list:
        orientation = orientation[0]

    if orientation == 1:
        pass
    elif orientation == 2:
        image = cv2.flip(image, 0)
    elif orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        image = cv2.flip(image, 1)
    elif orientation == 5:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image 

#1- Skin detection
def skinDetection(src):
    
    
    
    # Convert to CMYK space color and get M chanel
    src = fix_orientation(src,6)
    skinMask=get_skin_mask_cmyk(src)
   
    #rows,cols=skinMask.shape
    
    # 1.3- remove the false positive errors, 
    #the largest connected component is found using the
    #standard run-length encoding technique

  
    #labels = label(skinMask, return_num=False)

    #maxCC_withbcg = labels == np.argmax(np.bincount(labels.flat))
    #maxCC_nobcg = labels == np.argmax(np.bincount(labels.flat, weights=skinMask.flat))

    #skinMask=  np.where(maxCC_nobcg)

    #1.4- remove the false negative errors, 
    # image opening operation is performed (erode and dilate)
    
    kernelSize = (11, 11)
    anchor = (-1, -1)
    iterations = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernelSize)
    skinMask = cv2.erode(skinMask, kernel,anchor =anchor, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel,anchor =anchor, iterations=2)
    
    #blur the mask to help remove noise, then apply the
    #mask to the frame
    ksize = (3, 3)
    skinMask=cv2.GaussianBlur(skinMask,ksize,0)

    # Perform another OTSU threshold and search for biggest contour
    ret, skinMask = cv2.threshold(skinMask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(skinMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)

    # Create a new mask for the result image
    h, w = src.shape[:2]
    skinMask = np.zeros((h, w), np.uint8)

    # Draw the contour on the new mask and perform the bitwise operation
    cv2.drawContours(skinMask, [cnt],-1, 255, -1)
    
    # extract ROI
     #2- extract ROI
    (thresh, skinMask) = cv2.threshold(skinMask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    points = np.argwhere(skinMask>thresh) # find where the black pixels are
    points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
    x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
    skinMask = skinMask[y:y+h][:, x:x+w] # create a cropped region of the gray image
    src = src[y:y+h][:,x:x+w] # create a cropped region of the gray image

    
    skinMask = np.where(skinMask > thresh, 1, 0).astype(np.uint8)
    
    
    #apply mask on image
    img_segmented = cv2.bitwise_and(src,src,mask= skinMask)
    return img_segmented


def normalise (img,mean,std):
    normed = (img - np.mean(img))/(np.std(img))    
    return (normed)
    


def matCos(source):
    rows,cols = source.shape

    result =  np.zeros((cols, rows), np.float32)
    for  r in range( 0,  rows):
        for  c in range(0, cols):
            result[r, c]= math.cos(source[r, c])
        
    return result
    


def meshGrid(size) :
    l = (size * 2) + 1
    value = - size
    result =  np.zeros((l, l), np.float32)

    for c in range(0,l) :
        for  r in range (0,l):
            result[r, c]= value
        
        value=value+1
        
    return result
    



def  calculateMean(m) :
    summ = 0
    for aM in m:
        summ= summ + aM
        
    result =  summ / m.shape[0]
    
    return result
    


def getMedianFrequency(image):
    values = []
    rows,cols = image.shape
    for r  in range(0,rows):
        for c in range(0, cols):
            value = image[r, c]
            if (value > 0) :
                values.append(value)
                
    values.sort()
    size =len(values)
    median = 0

    if (size > 0) :
        halfSize = size //2
        if ((size % 2) == 0) :
            median = (values[halfSize - 1]+ values[halfSize]) / 2.0
        else:
            median = values[halfSize]

        
    return median
            
        
    
def gaussianKerne2l(kSize, Sigma):
    gauss = cv2.getGaussianKernel(kSize, Sigma)
    kernel= gauss * gauss.T
    return kernel

def gaussianKernel(kSize, Sigma):
    kernelX = cv2.getGaussianKernel(kSize, Sigma)
    kernelY = cv2.getGaussianKernel(kSize, Sigma)
    
   #print('kernelX',kernelX.shape,kernelX.dtype)
    #print('kernelY',kernelY.shape,kernelY.dtype)

    #kernel = np.zeros((kSize, kSize), np.float32)
    #kernel= np.outer(kernelX, kernelY.transpose())
    #kernel=cv2.multiply(kernelX.transpose(), kernelY.transpose())
    fill_third= np.zeros((kSize, kSize), np.float32)
    #fill_third[:]=0.0
    kernel= cv2.gemm(kernelX, np.transpose(kernelY), 1,fill_third, 0, 0)
    #print('kernel',kernel.shape,kernel.dtype)
    #kernel=cv2.multiply(kernelX, np.transpose(kernelY))
    return kernel


def atan2(padded, src1,src2):
    dst= np.zeros(padded.shape, np.float32)
    height = src1.shape[0]
    width = src2.shape[1]
    
    for y  in range(0,height) :
        for x in  range(0,width):
            dst[y, x]= cv2.fastAtan2(float(src1[y, x]), float(src2[y, x]))
            
    return dst




def snapShotMask(rows, cols, padding) :
        
    #Some magic numbers. We have no idea where these come from?!
       #int maskWidth = 260
    #int maskHeight = 160


    center = (int(cols / 2), int(rows / 2))
    axes = (int(cols / 2 - padding), int(rows / 2 - padding))
    scalarWhite = (255, 255, 255)
    scalarBlack = (0, 0, 0)
    thickness = -1
    lineType = 8

    mask = np.zeros((rows,cols), np.uint8) 
    mask[:]=0
    mask=cv2.ellipse(mask, center, axes, 0.0, 0.0, 360.0, scalarWhite, thickness, lineType, 0)
    
    return mask
    
# Apply padding to the image.
# create a border around the image like a photo frame
def imagePadding(source, blockSize):
    
    height, width = source.shape
    #print('source before',source.shape,source.dtype)

    bottomPadding = 0
    rightPadding = 0

    if ((width % blockSize)!= 0) :
        bottomPadding = blockSize - (width % blockSize)

    if ((height % blockSize) != 0) :
        rightPadding = blockSize - (height % blockSize)

    # Using cv2.copyMakeBorder() method
    source = cv2.copyMakeBorder(source, 0, bottomPadding, 0, rightPadding, cv2.BORDER_CONSTANT, 0)
    #print('source after padd:',source.shape,source.dtype)
    return source

 
def ridge_segment2(source,blockSize,threshold):
    source=normalise(source,0,1)
    rows, cols = source.shape
    new_rows =  int(blockSize * math.ceil((float(rows))/(float(blockSize))))
    new_cols =  int(blockSize * math.ceil((float(cols))/(float(blockSize))))
    
    padded_img = np.zeros((new_rows,new_cols))
    stddevim = np.zeros((new_rows,new_cols))
    
    padded_img[0:rows][:,0:cols] = source
    
     
    for i in range(0,new_rows,blockSize):
        for j in range(0,new_cols,blockSize):
            block = padded_img[i:i+blockSize][:,j:j+blockSize]
            stddevim[i:i+blockSize][:,j:j+blockSize] = np.std(block)*np.ones(block.shape)
    
    stddevim = stddevim[0:rows][:,0:cols]
                    
    mask = stddevim > threshold
    
    
    mean_val = np.mean(source[mask])
    
    std_val = np.std(source[mask])
    
    normim = (source - mean_val)/(std_val)
    #print(mean_val,std_val)
    
    return(normim,mask)




def calculateFrequency(block, blockOrientation, windowSize, minWaveLength, maxWaveLength):
    #print('block: ',block.shape,block.dtype)
    #print('block val: ',np.unique(block))
    
    rows,cols = block.shape
    orientation =copy.deepcopy(blockOrientation)
    orientation= cv2.multiply(orientation,2.0)
    
    orientations= orientation.flatten()
    orientLength =orientations.shape[0]
    
    sinOrient = np.zeros(shape=orientLength, dtype=np.float64)
    cosOrient = np.zeros(shape=orientLength, dtype=np.float64)
    
    for i in range(1,orientLength):
        sinOrient[i] = math.sin(orientations[i])
        cosOrient[i] = math.cos(orientations[i])
        
        
    # Find mean orientation within the block. This is done by averaging the
    # sines and cosines of the doubled angles before reconstructing the
    # angle again.  This avoids wraparound problems at the origin.
        
    
    orient = cv2.fastAtan2(float(calculateMean(sinOrient)), float(calculateMean(cosOrient))) / float(2.0)

    #rotate the image block so that the ridges are vertical
    #  Mat rotated = new Mat(rows, cols, CvType.CV_32FC1);
    center = (cols / 2, rows / 2)
    rotateAngle = ((orient / math.pi) * (180.0)) + 90.0
    rotateScale = 1.0
    rotatedSize = (cols, rows)
    rotateMatrix = cv2.getRotationMatrix2D(center, rotateAngle, rotateScale)
    rotated= cv2.warpAffine(block, rotateMatrix, rotatedSize, flags = cv2.INTER_NEAREST)
    
     
    #print('rotated: ',rotated.shape,rotated.dtype)
    #print('rotated val: ',np.unique(rotated))

    # crop the image so that the rotated image does not contain any invalid regions
    # this prevents the projection down the columns from being mucked up
    cropSize = int(np.fix(rows / np.sqrt(2)))
    #offset = int(np.fix((rows - cropSize) /2.0) - 1)
    offset = int(np.fix((rows - cropSize) /2.0)-1)
    cropped = rotated[offset:offset + cropSize][:, offset:offset + cropSize]
  
    #print('cropped: ',cropped.shape,cropped.dtype)
    #print('cropped val: ',np.unique(cropped))
    
   
    #get sums of columns
    #  Mat proj = new Mat(1, cropped.cols(), CvType.CV_32FC1)
    
    proj =   np.zeros((1,cropped.shape[1]), np.float32)
    for c in range(1,cropped.shape[1]):
        sum = 0
        for  r in range(1,cropped.shape[1]):
            sum = sum + cropped[r, c]

        proj[0, c] = sum
        
    #proj = np.sum(cropped,axis = 0)
   
    #find peaks in projected grey values by performing a grayScale
    #dilation and then finding where the dilation equals the original values.
    dilateKernel = np.zeros((windowSize, windowSize), np.float32)
    dilateKernel[:]=1.0
    # dilate = np.zeros((1, cropped.shape[1]), np.float32)

    dilate= cv2.dilate(proj, dilateKernel,(-1,-1),iterations = 1)
    #dilate=cv2.dilate(proj, dilateKernel, (-1, -1), 1, cv2.BORDER_CONSTANT, (0.0,0.0,0.0,0.0))

    # print('projr: ',proj.shape,proj.dtype)
    # print('proj val: ',np.unique(proj))

    projMean = cv2.mean(proj)[0]

    #print(projMean)

    ROUND_POINTS = 1000
    maxind = []
    for i in range(0,cropped.shape[1]) :

        projValue = proj[0, i]
        dilateValue = dilate[0, i]

        #round to maximize the likelihood of equality
        projValue = float(np.fix(projValue * ROUND_POINTS) / ROUND_POINTS)
        dilateValue = float(np.fix(dilateValue * ROUND_POINTS) / ROUND_POINTS)

        if ((dilateValue == projValue) and (projValue > projMean)) :
            maxind.append(i)

    #determine the spatial frequency of the ridges by dividing the distance between
    #the 1st and last peaks by the (No of peaks-1). If no peaks are detected
    #or the wavelength is outside the allowed bounds, the frequency image is set to 0
    #Mat result = new Mat(rows, cols, CvType.CV_32FC1, Scalar.all(0.0))
    result =  np.zeros((rows,cols), np.float32)
    result[:]=0.0
    peaks = len(maxind)
    if (peaks >= 2) :
        waveLength = (maxind[peaks - 1] - maxind[0]) / (peaks - 1)
        #print('waveLength ',waveLength)
        if ((waveLength >= minWaveLength ) and (waveLength <= maxWaveLength)):
            result =  np.zeros((rows,cols), np.float32)
            result[:]=1.0/ waveLength

    #print('result', result.shape, np.unique(result))

    return result

def getSkeletonImage(src, rows, cols):
    
    #step 1: get ridge segment by padding then do block process
    blockSize = 24 #16
    threshold = 0.05 #0.1
    padded = imagePadding(src, blockSize)
    imgRows,imgCols= padded.shape
    
    matRidgeSegment, segmentMask=ridgeSegment(padded, blockSize, threshold)
    #return matRidgeSegment
    
    
    #step 2: get ridge orientation
    gradientSigma = 1 # 1
    blockSigma = 13 #7
    orientSmoothSigma = 15 #7
    matRidgeOrientation = ridgeOrientation(padded, matRidgeSegment, gradientSigma, blockSigma, orientSmoothSigma)
  
    
    #step 3: get ridge frequency
    fBlockSize = 36 #38
    fWindowSize = 5
    fMinWaveLength = 5
    fMaxWaveLength = 25 #15
    
    matFrequency, medianFreq= ridgeFrequency(padded, matRidgeSegment, segmentMask, matRidgeOrientation, fBlockSize, fWindowSize, fMinWaveLength, fMaxWaveLength)
   
    #step 4: get ridge filter
    filterSize = 1.9 #0.65
    matRidgeFilter, padding = ridgeFilter(padded,matRidgeSegment, matRidgeOrientation, matFrequency, filterSize, filterSize, medianFreq)
    paddingSize = padding # globalb variable use later for filterMinutiae
    
  
    #step 5: enhance image after ridge filter
    matEnhanced =enhancement(matRidgeFilter, blockSize, rows, cols, padding)
    
    return (padding, matEnhanced)
    

def enhancement(source, blockSize, rows, cols, padding):
    
    MatSnapShotMask = snapShotMask(rows, cols, padding)
   
    paddedMask = imagePadding(MatSnapShotMask, blockSize)
    

    paddedMask=paddedMask.astype(np.float32)
    #print('MatSnapShotMask',MatSnapShotMask.shape,MatSnapShotMask.dtype)
    #print('paddedMask',paddedMask.shape,paddedMask.dtype)
    #print('source',source.shape,source.dtype)



    # apply the original mask to get rid of extras
    result=cv2.multiply(source, paddedMask,1.0)
    #result= result.astype(np.uint8)
    #result= source* paddedMask

    # apply binary threshold
    result= cv2.threshold(result, 0, 255, cv2.THRESH_BINARY)[1]
    
    return result
    

    

# RIDGESEGMENT - Normalises fingerphotos image and segments ridge region
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
# Usage:   [normim, mask, maskind] = ridgesegment(im, blksze, thresh)
#
# Arguments:   im     - Fingerphotos image to be segmented.
#              blksze - Block size over which the the standard
#                       deviation is determined (try a value 16 - 24).
#              thresh - Threshold of standard deviation to decide if a
#                       block is a ridge region (Try a value 0.05 - 0.2)
#
# Returns:     matRidgeSegment - Image where the ridge regions are renormalised to
#                                have zero mean, unit standard deviation.
#              segmentMask   - Mask indicating ridge-like regions of the image, 
#                             0 for non ridge regions, 1 for ridge regions.
#            
#
# Suggested values for a 500dpi fingerprint image:
#
#   [normim, mask] = ridgesegment(im, 24, 0.05)
#
# See also: RIDGEORIENT, RIDGEFREQ, RIDGEFILTER


def ridgeSegment(source,blockSize,threshold):
    
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



# RIDGEORIENT - Estimates the local orientation of ridges in a fingerprint
#
# Usage:  [orientim, reliability, coherence] = ridgeorientation(im, gradientsigma,...
#                                             blocksigma, ...
#                                             orientsmoothsigma)
#
# Arguments:  im                - A normalised input image.
#             gradientsigma     - Sigma of the derivative of Gaussian
#                                 used to compute image gradients.
#             blocksigma        - Sigma of the Gaussian weighting used to
#                                 sum the gradient moments.
#             orientsmoothsigma - Sigma of the Gaussian used to smooth
#                                 the final orientation vector field. 
#                                 Optional: if ommitted it defaults to 0
# 
# Returns:    orientim          - The orientation image in radians.
#                                 Orientation values are +ve clockwise
#                                 and give the direction *along* the
#                                 ridges.
#             reliability       - Measure of the reliability of the
#                                 orientation measure.  This is a value
#                                 between 0 and 1. I think a value above
#                                 about 0.5 can be considered 'reliable'.
#                                 reliability = 1 - Imin./(Imax+.001);
#             coherence         - A measure of the degree to which the local
#                                 area is oriented.
#                                 coherence = ((Imax-Imin)./(Imax+Imin)).^2;
#
# With a fingerprint image at a 'standard' resolution of 500dpi suggested
# parameter values might be:
#
#    [orientim, reliability] = ridgeorient(im, 1, 3, 3);
#
# See also: RIDGESEGMENT, RIDGEFREQ, RIDGEFILTER

def ridgeOrientation(padded,ridgeSegment, gradientSigma, blockSigma,orientSmoothSigma):
   
    rows, cols = ridgeSegment.shape

    # calculate image gradients
    kSize = int(np.fix(6 * gradientSigma))
    if (kSize % 2) == 0 :
        kSize=kSize+1


    kernel=gaussianKernel(kSize, gradientSigma)

    
    fXKernel = np.zeros((1,3), np.float32)
    fYKernel =np.zeros((3,1), np.float32)

    fXKernel[0,0]=-1
    fXKernel[0,1]= 0
    fXKernel[0,2]= 1

    fYKernel[0,0]=-1
    fYKernel[1,0]= 0
    fYKernel[2,0]= 1
    #fY,fX = np.gradient(kernel)


    fX=cv2.filter2D(kernel,  -1, fXKernel)
    fY=cv2.filter2D(kernel, -1, fYKernel)

    gX=cv2.filter2D(ridgeSegment,-1, fX)
    gY=cv2.filter2D(ridgeSegment,-1, fY)

    # covariance data for the image gradients
    gXX= cv2.multiply(gX, gX)
    gXY= cv2.multiply(gX, gY)
    gYY= cv2.multiply(gY, gY)

    #smooth the covariance data to perform a weighted summation of the data.
    kSize = int(np.fix(6 * blockSigma))
    if (kSize % 2) == 0  :
        kSize=kSize+1

    kernel=gaussianKernel(kSize, blockSigma)
    gXX= cv2.filter2D(gXX, -1,kernel)
    gYY= cv2.filter2D(gYY, -1,kernel)
    gXY= cv2.filter2D(gXY, -1,kernel)
    gXY= cv2.multiply(gXY, (2,0,0,0))


    #analytic solution of principal direction
    gXXMiusgYY= cv2.subtract(gXX, gYY)
    gXXMiusgYYSquared= cv2.multiply(gXXMiusgYY, gXXMiusgYY)
    gXYSquared= cv2.multiply(gXY, gXY)
    denom= cv2.add(gXXMiusgYYSquared, gXYSquared)
    denom = cv2.sqrt(denom)

    #sine and cosine of doubled angles
    sin2Theta= cv2.divide(gXY, denom)
    cos2Theta= cv2.divide(gXXMiusgYY, denom)

    #smooth orientations (sine and cosine)
    #smoothed sine and cosine of doubled angles
    kSize = int(np.fix(6 * orientSmoothSigma))
    if (kSize % 2) == 0  :
        kSize=kSize+1

    kernel = gaussianKernel(kSize, orientSmoothSigma)
    sin2Theta = cv2.filter2D(sin2Theta,-1,kernel)
    cos2Theta = cv2.filter2D(cos2Theta, -1,kernel)

    #calculate the result as the following, so the values of the matrix range [0, PI]
    result =atan2(padded,sin2Theta,cos2Theta)
    #result =np.arctan2(sin2Theta,cos2Theta)
    scale= math.pi / 360.0   
    result=cv2.multiply(result, scale)
    
    return result

# RIDGEFREQ - Calculates a ridge frequency image
#
# Function to estimate the fingerprint ridge frequency across a
# fingerprint image. This is done by considering blocks of the image and
# determining a ridgecount within each block by a call to FREQEST.
#
# Usage:
#  [freqim, medianfreq] =  ridgefreq(im, mask, orientim, blksze, windsze, ...
#                                    minWaveLength, maxWaveLength)
#
# Arguments:
#         im       - Image to be processed.
#         mask     - Mask defining ridge regions (obtained from RIDGESEGMENT)
#         orientim - Ridge orientation image (obtained from RIDGORIENT)
#         blksze   - Size of image block to use (say 32) 
#         windsze  - Window length used to identify peaks. This should be
#                    an odd integer, say 3 or 5.
#         minWaveLength,  maxWaveLength - Minimum and maximum ridge
#                     wavelengths, in pixels, considered acceptable.
# 
# Returns:
#         freqim     - An image  the same size as im with  values set to
#                      the estimated ridge spatial frequency within each
#                      image block.  If a  ridge frequency cannot be
#                      found within a block, or cannot be found within the
#                      limits set by min and max Wavlength freqim is set
#                      to zeros within that block.
#         medianfreq - Median frequency value evaluated over all the
#                      valid regions of the image.
#
# Suggested parameters for a 500dpi fingerprint image
#   [freqim, medianfreq] = ridgefreq(im,orientim, 32, 5, 5, 15);
#

# See also: RIDGEORIENT, FREQEST, RIDGESEGMENT


def ridgeFrequency(padded, ridgeSegment, segmentMask, ridgeOrientation, blockSize, windowSize, minWaveLength,maxWaveLength):
    
    frequencies=  np.zeros(padded.shape, np.float32)
    rows,cols = ridgeSegment.shape

    for y  in range (0,rows - blockSize,blockSize):
        for x  in range (0, cols - blockSize,blockSize):
            y1=y
            y2=y1+blockSize
            x1=x
            x2=x1+blockSize
            blockSegment = ridgeSegment[y1:y2][: , x1:x2]
            blockOrientation = ridgeOrientation[y1:y2][:,x1:x2]
            #print(blockOrientation.shape,np.unique(blockOrientation))

            frequency = calculateFrequency(blockSegment, blockOrientation, windowSize, minWaveLength, maxWaveLength)
         
            # draw frequency on matFrequency
            # frequency.copyTo(matFrequency.rowRange(y, y + blockSize).colRange(x, x + blockSize))
            frequencies[y1:y2][:,x1:x2]=frequency
            

    segmentMask=segmentMask.astype(np.float32)
    #print('matFrequency before: ',matFrequency.shape,matFrequency.dtype)
    #print('matFrequency before val: ',np.unique(matFrequency))
    #mask out frequencies calculated for non ridge regions
    frequencies =cv2.multiply(frequencies, segmentMask, 1.0)
    
   
    #frequencies =frequencies*segmentMask
  
    #frequencies_1d = np.reshape(frequencies,(1,rows*cols))
    #ind = np.where(frequencies_1d>0)
    
    #ind = np.array(ind)
    #ind = ind[1,:]    
    
    #non_zero_elems_in_frequencies = frequencies_1d[0][ind]  
    #print(np.unique(non_zero_elems_in_frequencies))
    
    #meanfreq = np.mean(non_zero_elems_in_frequencies)
    #medianFrequency = np.median(non_zero_elems_in_frequencies) 

    #print(frequencies.shape,frequencies.dtype)

    # ind median frequency over all the valid regions of the image.
  
    medianFrequency = getMedianFrequency(frequencies)
    # the median frequency value used across the whole fingerprint gives a more satisfactory result
    frequencies= cv2.multiply(segmentMask,medianFrequency, 1.0)  

    #segmentMask=segmentMask.astype(np.uint8)
    #print('matFrequency after: ',matFrequency.shape,matFrequency.dtype)
    #print('matFrequency after val: ',np.unique(matFrequency))
    
    return (frequencies,medianFrequency)





# RIDGEFILTER - enhances fingerprint image via oriented filters
#
# Function to enhance fingerprint image via oriented filters
#
# Usage:
#  newim =  ridgefilter(im, orientim, freqim, kx, ky, showfilter)
#
# Arguments:
#         im       - Image to be processed.
#         orientim - Ridge orientation image, obtained from RIDGEORIENT.
#         freqim   - Ridge frequency image, obtained from RIDGEFREQ.
#         kx, ky   - Scale factors specifying the filter sigma relative
#                    to the wavelength of the filter.  This is done so
#                    that the shapes of the filters are invariant to the
#                    scale.  kx controls the sigma in the x direction
#                    which is along the filter, and hence controls the
#                    bandwidth of the filter.  ky controls the sigma
#                    across the filter and hence controls the
#                    orientational selectivity of the filter. A value of
#                    0.5 for both kx and ky is a good starting point.
#         showfilter - An optional flag 0/1.  When set an image of the
#                      largest scale filter is displayed for inspection.
# 
# Returns:
#         newim    - The enhanced image
#
# See also: RIDGEORIENT, RIDGEFREQ, RIDGESEGMENT


import scipy

def ridgeFilter(padded,ridgeSegment, orientation, frequency, kx, ky, medianFreq):
    result=  np.zeros(padded.shape, np.float32)
    angleInc = 3
    rows,cols = ridgeSegment.shape
    filterCount = 180 // angleInc
    filters = []
    sigmaX = kx / medianFreq
    sigmaY = ky / medianFreq


    # mat refFilter = exp(-(x. ^ 2 / sigmaX ^ 2 + y. ^ 2 / sigmaY ^ 2) / 2). * cos(2 * pi * medianFreq * x)
    size = int(np.fix(3 * max(sigmaX, sigmaY)))
    size =size if ((size % 2) == 0)  else (size + 1)
    length = (size * 2) + 1
    x = meshGrid(size)
    y = np.transpose(x)

    xSquared=cv2.multiply(x, x)
    ySquared=cv2.multiply(y, y)
    xSquared=cv2.divide(xSquared, sigmaX * sigmaX)
    ySquared=cv2.divide(ySquared, sigmaY * sigmaY)


    refFilterPart1=cv2.add(xSquared, ySquared)
    refFilterPart1=cv2.divide(refFilterPart1,-2.0)
    refFilterPart1= cv2.exp(refFilterPart1)


    refFilterPart2= cv2.multiply(x, 2 * math.pi * medianFreq)
    refFilterPart2 = matCos(refFilterPart2)

    refFilter=cv2.multiply(refFilterPart1, refFilterPart2)

    #Generate rotated versions of the filter.  Note orientation
    # image provides orientation *along* the ridges, hence +90
    # degrees, and the function requires angles +ve anticlockwise, hence the minus sign.

    center = (length // 2, length // 2)
    rotatedSize = (length, length)
    rotateScale = 1.0

    for i in range(0, filterCount):
        rotateAngle = - (i * angleInc)
        rotateMatrix = cv2.getRotationMatrix2D(center, rotateAngle, rotateScale)
        rotated=cv2.warpAffine(refFilter, rotateMatrix, rotatedSize, cv2.INTER_LINEAR)
        #filters[i] = rotated
        filters.append(rotated)


    #convert orientation matrix values from radians to an index value
    # that corresponds to round(degrees/angleInc)
    orientIndexes=cv2.multiply(orientation, float(filterCount) / math.pi, 1.0)

    orientThreshold = np.zeros(orientation.shape, np.uint8)
    orientThreshold[:]=0

    orientIndexes=orientIndexes.astype(np.uint8)
    #print('orientIndexes',orientIndexes.shape,orientIndexes.dtype)
    #print('orientThreshold',orientThreshold.shape,orientThreshold.dtype)

    orientMask= cv2.compare(orientIndexes, orientThreshold, cv2.CMP_LT)
    orientIndexes=cv2.add(orientIndexes, filterCount,  orientMask)

    orientMask =  np.zeros(orientation.shape, np.uint8)
    orientMask[:]=0
    orientThreshold = np.zeros(orientation.shape, np.uint8)
    orientThreshold[:]=filterCount


    orientMask=cv2.compare(orientIndexes,orientThreshold, cv2.CMP_GE)
    orientIndexes=cv2.subtract(orientIndexes, filterCount, orientMask)

    #finally, find where there is valid frequency data then do the filtering


    for r in range(0,rows):
        for c in range (0 ,cols):
            if (frequency[r, c] > 0
                    and r > (size + 1)
                    and r < (rows - size - 1)
                    and c > (size + 1)
                    and c < (cols - size - 1)) :
               
                orientIndex = int(orientIndexes[r, c])
                subSegment = ridgeSegment[r - size - 1:r + size][:, c - size - 1: c + size]
                value=cv2.multiply(subSegment, filters[orientIndex])
                sum = cv2.sumElems(value)[0]
                result[r, c]= sum




    return (result,size)




def x(a, i, j):
    try:
        if(a==1):
            return B[i + 1][j]
        elif(a==2):
            return B[i + 1][j + 1]
        elif(a==3):
            return B[i][j + 1]
        elif(a==4):  
            return B[i - 1][j + 1]
        elif(a==5):
            return B[i - 1][j]
        elif(a==6):
            return B[i - 1][j - 1]
        elif(a==7):   
            return B[i][j - 1]
        elif(a==8):  
            return B[i + 1][j - 1]

    except :
        return False

    return False
    

def G1(i, j) :
    X = 0
    for q in range(1, 5):
        if ((not x(2 * q - 1, i, j)) and (x(2 * q, i, j) or x(2 * q + 1, i, j))):
            X=X+1
    
    return X == 1


def G2(i,j):
    m = min(n1(i, j), n2(i, j))
    return (m == 2 or m == 3)


def n1(i, j) :
    r = 0
    for q in range(1,5):
        if (x(2 * q - 1, i, j) or x(2 * q, i, j)): 
            r=r+1
    return r


def n2(i, j):
    r = 0
    for q in range(1,5):
        if (x(2 * q, i, j) or x(2 * q + 1, i, j)):
            r=r+1
    return r


def G3(i, j) :
    return (x(2, i, j) or x(3, i, j) or (not x(8, i, j))) and x(1, i, j)


def G3_(i,j) :
    return (x(6, i, j) or x(7, i, j) or (not x(4, i, j))) and x(5, i, j)

def neighbourCount(i,j):
    cn = 0
    for a in range(1,9):
        if (x(a, i, j)):
            cn=cn+1
    return cn

#follow ridge recursively and remove if shorter than minimumlength
def removeEnding(i, j,minimumLength):
    if (minimumLength < 0):
        return True
    if (neighbourCount(i, j) > 1):
        return False
    
    B[i][j] = False
    if (neighbourCount(i, j) == 0):
        return False
    index = 0
    for a in range(1, 9) :
        if (x(a, i, j)):
            index = a
            break
        
    
    _i = i
    _j = j

    if(index==1):
        _i = i + 1
    elif (index==2):
        _i = i + 1
        _j = j + 1

    elif (index==3):
        _j = j + 1

    elif (index==4):
        _i = i - 1
        _j = j + 1
    elif (index==5):
        _i = i - 1       
    elif (index==6):
        _i = i - 1
        _j = j - 1
    elif (index==7):
        _j = j - 1
    elif (index==8):
        _i = i + 1
        _j = j - 1
           
    
    ok = removeEnding(_i, _j, minimumLength - 1)
    if (ok):
        B[i][j] = True
    return ok

def thinning2(img):
    img1 = img.copy()
 
    # Structuring Element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    # Create an empty output image to hold values
    thin = np.zeros(img.shape,dtype='uint8')

    # Loop until erosion leads to an empty set
    while (cv2.countNonZero(img1)!=0):
        # Erosion
        erode = cv2.erode(img1,kernel)
        # Opening on eroded image
        opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel)
        # Subtract these two
        subset = erode - opening
        # Union of all previous sets
        thin = cv2.bitwise_or(subset,thin)
        # Set the eroded image for next iteration
        img1 = erode.copy()
    
    return thin


class Thinning(object):
    def __init__(self):
        self.B = []

    def doJaniThinning(self,Image):
    
        self.B = np.zeros(Image.shape, np.uint8)
        #Inverse of B
        B_ = np.zeros(Image.shape, np.uint8)

        self.B= Image>10 #not a mistake, in matlab first invert and then morph
        prevB = np.zeros(Image.shape, np.uint8)

        maxIter = 1000

        for iter in range(0,maxIter):
            #Assign B to prevB
            prevB[:]= self.B

            #Iteration #1
            for i in range(0, Image.shape[0]):
                for  j  in range(0,Image.shape[1]):
                    B_[i][j] = (not(self.B[i][j] and G1(i, j) and G2(i, j) and G3(i, j))) and self.B[i][j]

            #Assign result of iteration #1 to B, so that iteration #2 will see the results
            self.B[:]= B_

            #Iteration #2
            for i in range(0, Image.shape[0]):
                for  j  in range(0,Image.shape[1]):
                    B_[i][j] = (not(self.B[i][j] and G1(i, j) and G2(i, j) and G3(i, j))) and self.B[i][j]

            # Assign result of Iteration #2 to B
            self.B[:]= B_

            #stop when it doesn't change anymore
            convergence = True
            for i in range(0,Image.shape[0]):
                convergence =convergence and  np.array_equal(self.B[i], prevB[i])
            if (convergence):
                break


        # remove ridge endings shorter than minimumRidgeLength
        minimumRidgeLength = 5
        for i in range(0,Image.shape[0]):
            for j in range (0,Image.shape[1]):
                if (self.B[i][j] and neighbourCount(i, j) == 1):
                    removeEnding(i, j, minimumRidgeLength)


        r = np.zeros(Image.shape, np.uint8)

        for i in range(0,Image.shape[0]):
            for  j in range(0,Image.shape[1]):
                if (self.B[i][j]):
                    r[i, j]= 255

        return r

        
def thinning(img):
    thinned= cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
    t = Thinning()
    thinned = t.doJaniThinning(thinned)
    
    return thinned
   


def equalize_histogram(img):
    return cv2.equalizeHist(img)

def enhance_image_photo_finger(img):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_eq=cv2.equalizeHist(img)
    rows,cols=image_eq.shape
    floated = image_eq.astype(np.float32)
    #2- skeletization
    padding, skeleton = getSkeletonImage(floated,rows,cols)
    return (padding,cv2.normalize(np.uint8(skeleton), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=0))




#Design HashSet in python
#checking the values and will return the output class
class checkingvalues:
    #initialization function which has list mathfun
    def __init__(self):
        self.mathfun=[]

    #update vales function
    def update(self, key):
        found=False
        for i,k in enumerate(self.mathfun):
            if key==k:
                self.mathfun[i]=key
                found=True
                break
        if not found:
            self.mathfun.append(key)

    #get values function
    def get(self, key):
        for k in self.mathfun:
            if k==key:
                return True
        return False
    #remove values function
    def remove(self, key):
        for i,k in enumerate(self.mathfun):
             if key==k:
                del self.mathfun[i]


# class HashSet main class
class HashSet:
    #Initialization function
    def __init__(self):
        self.key_space = 2096
        self.hash_table=[checkingvalues() for i in range(self.key_space)]
    
    def hash_values(self, key):
        hash_key=key%self.key_space
        return hash_key
    #add function
    def add(self, key):
        self.hash_table[self.hash_values(key)].update(key)
    #remove function
    def remove(self, key):
        self.hash_table[self.hash_values(key)].remove(key)
    #contains function
    def contains(self, key):
         return self.hash_table[self.hash_values(key)].get(key)
    def display(self):
        ls=[]
        for i in self.hash_table:
            if len(i.mathfun)!=0:ls.append(i.mathfun[0])
        print(ls)
  
    
        


from enum import Enum
 

class Minutiae :
    BIFURCATION_LABEL = 1
    RIDGE_ENDING_LABEL = 0
    def __init__(self,x,y,target_type):
        self.x = x
        self.y = y
        self.type = target_type
    
    def euclideanDistance(self,m):
        return np.sqrt(np.power(self.x - m.x, 2) + np.power(self.y - m.y, 2))
    
    class Type(Enum):
        BIFURCATION = 0
        RIDGEENDING = 1

def removeMinutiae(minDistance, source, target):
        toBeRemoved = set()
        check= source.copy() 
       
        for m in source :
            if (m in toBeRemoved):
                print("continue")
                continue
            ok = True
            print("ok")
           
            check.remove(m)
            print(len(check))
            for m2 in check: 
                if (m.euclideanDistance(m2) < minDistance) :
                    ok = False
                    toBeRemoved.add(m2)
                
            
            if (ok==True):
                target.add(m)
            else:
                toBeRemoved.add(m)
        
        for m in target:
            print(m)
        return target
        
def filterMinutiae(src,skeleton,paddingSize) :
    rows,cols= skeleton.shape
    
    mask= snapShotMask3(rows,cols, paddingSize + 5)
    ridgeEnding = set()
    bifurcation = set()
    filtered = set()
    for m in src :
        if (mask[m.y, m.x]> 0):  #filter out borders
            if (m.type == Minutiae.Type.BIFURCATION):
                ridgeEnding.add(m)
            else:
                bifurcation.add(m)
        
    
            

    minDistance = 5
   
    try:
        filtered=removeMinutiae(minDistance, ridgeEnding, filtered)
        print("ok for next")
        filtered=removeMinutiae(minDistance, bifurcation, filtered)
    except Exception as e:
        print(e)
    return filtered
    


def detectMinutiae(skeleton, border,orb,padding) :
        minutiaeSet = set()
        rows,cols=skeleton.shape
        for c in range(border,cols - border):
            for r in range(border,rows - border):
                point = skeleton[r, c]
                if (point != 0):
                    cn = neighbourCount3(skeleton, r, c)
                    if (cn == 1):
                        minutiaeSet.add(Minutiae(c, r, Minutiae.Type.RIDGEENDING))
                    elif (cn == 3):
                        minutiaeSet.add(Minutiae(c, r, Minutiae.Type.BIFURCATION))
                        
        
       
      
        filteredMinutiae = filterMinutiae(minutiaeSet, skeleton,padding)
        for m in filteredMinutiae:
            print(m)
        
       
        result=cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
        red = [255, 0, 0]
        green = [0, 255, 0]
        for m in filteredMinutiae: 
            
            if (m.type == Minutiae.Type.BIFURCATION):
                color = green
            else:
                color = red
                
            result[m.y, m.x]=  color
            result[m.y, m.x - 1]= color
            result[m.y, m.x + 1]= color
            result[m.y - 1, m.x]= color
            result[m.y + 1, m.x] = color
        
        keypoints=minutiaeToKeyPoints( skeleton, filteredMinutiae)
        kp, des = orb.compute(skeleton, keypoints)
      
        return (result,kp, des)
    
    
def getMinutiaeAngle(skeleton,m) :

        try :
            direction = followRidge(skeleton, 5, m.x, m.y, m.x, m.y)
        except ValueError as e:  
            print(e)
            return -1
        
        length = np.sqrt(np.power(direction.x - m.x, 2) + np.power(direction.y - m.y, 2))
        cosine = (direction.y - m.y) / length
        angle = float (math.acos(cosine))
        if (direction.x - m.x < 0):
            angle = -angle + 2 * float(math.pi)
        return angle
    

def followRidge(skeleton, length, currentX, currentY, previousX, previousY) :
        if (length == 0):
            return Minutiae(currentX, currentY, Minutiae.Type.RIDGEENDING);
        if (currentY >= skeleton.shape[0] - 1 or currentY == 0 or currentX >= skeleton.shape[1] - 1 or currentX == 0):
            raise ValueError('out of bounds.')

        _x = currentX - 1
        _y = currentY - 1
        if (followRidgeCheck(skeleton, _x, _y, previousX, previousY)):
            return followRidge(skeleton, length - 1, _x, _y, currentX, currentY);
        _x = currentX - 1
        _y = currentY
        if (followRidgeCheck(skeleton, _x, _y, previousX, previousY)):
            return followRidge(skeleton, length - 1, _x, _y, currentX, currentY)
        _x = currentX - 1
        _y = currentY + 1
        if (followRidgeCheck(skeleton, _x, _y, previousX, previousY)):
            return followRidge(skeleton, length - 1, _x, _y, currentX, currentY)
        _x = currentX
        _y = currentY - 1
        if (followRidgeCheck(skeleton, _x, _y, previousX, previousY)):
            return followRidge(skeleton, length - 1, _x, _y, currentX, currentY)
        _x = currentX
        _y = currentY + 1
        if (followRidgeCheck(skeleton, _x, _y, previousX, previousY)):
            return followRidge(skeleton, length - 1, _x, _y, currentX, currentY)
        
        _x = currentX + 1
        _y = currentY - 1
        if (followRidgeCheck(skeleton, _x, _y, previousX, previousY)):
            return followRidge(skeleton, length - 1, _x, _y, currentX, currentY)
        
        _x = currentX + 1
        _y = currentY
        if (followRidgeCheck(skeleton, _x, _y, previousX, previousY)):
            return followRidge(skeleton, length - 1, _x, _y, currentX, currentY)
        
        _x = currentX + 1
        _y = currentY + 1
        return followRidge(skeleton, length - 1, _x, _y, currentX, currentY);
    

def followRidgeCheck(skeleton, x, y, previousX, previousY) :
        if (x == previousX and y == previousY):
            return False
        return skeleton[y, x] != 0

    
    
    
def minutiaeToKeyPoints(skeleton,minutiae):
    index = 0
    size = 1
    angle = -1
    response = 1
    octave = 1
    return [ cv2.KeyPoint(x=m.x,y= m.y, _size=size,_angle=getMinutiaeAngle(skeleton, m) if m.type == Minutiae.Type.RIDGEENDING else angle,_response=response,
                                        _octave=octave,_class_id=Minutiae.RIDGE_ENDING_LABEL if m.type == Minutiae.Type.RIDGEENDING else Minutiae.BIFURCATION_LABEL ) for m in minutiae ] 
    
    
def neighbourCount3(skeleton, row, col) :
        cn = 0
        if (skeleton[row - 1, col - 1]!= 0) : 
            cn=cn+1
        if (skeleton[row - 1, col]!= 0) :
            cn=cn+1
        if (skeleton[row - 1, col + 1] != 0) :
            cn=cn+1
        if (skeleton[row, col - 1]!= 0):
            cn=cn+1
        if (skeleton[row, col + 1]!= 0) :
            cn=cn+1
        if (skeleton[row + 1, col - 1] != 0) :
            cn=cn+1
        if (skeleton[row + 1, col] != 0):
            cn=cn+1
        if (skeleton[row + 1, col + 1] != 0):
            cn=cn+1
        return cn
    
def snapShotMask3(rows, cols, padding):
        
    #Some magic numbers. We have no idea where these come from?!
    # maskWidth = 260;
    # maskHeight = 160;


    center = (cols // 2, rows // 2)
    axes = (cols // 2 - padding, rows // 2 - padding)
    scalarWhite = (255, 255, 255)
    scalarBlack = (0, 0, 0)
    thickness = -1
    lineType = 8
    mask = np.zeros((rows,cols), np.uint8) 
    mask[:]=0
    mask=cv2.ellipse(mask, center, axes, 0, 0, 360, scalarWhite, thickness, lineType, 0) 
    return mask
    