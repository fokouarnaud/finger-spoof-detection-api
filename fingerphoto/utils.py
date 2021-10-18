import cv2
import numpy as np
from .utils2 import *
from .fingerprint_feature_extractor import *

from .constants import IMAGES_PATH, IMAGES_PATH_GALLERY, IMAGES_PATH_PROBE

def enhance_image_target(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #img = img[250:950, 900:1912]

    rows, cols = np.shape(image)
    aspect_ratio = np.double(rows)/np.double(cols)

    new_rows = 612             # randomly selected number
    new_cols = new_rows/aspect_ratio

    image = cv2.resize(image,(np.int(new_cols),np.int(new_rows)))
    
  
    image_eq=cv2.equalizeHist(image)
    rows,cols=image_eq.shape
    floated = image_eq.astype(np.float32)
    #2- skeletization
    padding, skeleton = getSkeletonImage(floated,rows,cols)
    return (padding,cv2.normalize(np.uint8(skeleton), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=0))
  
 
def get_feature_keypoint_and_descriptor_old(image,orb,padding,border=1):
    kp, des = orb.detectAndCompute(image, None)
    #print('kp',kp)
    #print('des',des)
    return (kp, des)

def get_feature_keypoint_and_descriptor_old2(image,orb,padding,border=1):
    # Harris corners
    harris_corners = cv2.cornerHarris(image, 3, 3, 0.04)
    harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    threshold_harris = 125
    # Extract keypoints
    keypoints = []
    for x in range(0, harris_normalized.shape[0]):
        for y in range(0, harris_normalized.shape[1]):
            if harris_normalized[x][y] > threshold_harris:
                keypoints.append(cv2.KeyPoint(y, x, 1))

    # Compute descriptors
    _, des = orb.compute(image, keypoints)

    return (keypoints, des)

def minutiaeToKeyPoints_old(FeaturesTerminations, FeaturesBifurcations):
    terminaison_id=1
    bifurcation_id=2
    result = []
    size=1
    response = 1
    octave = 1
    kps_term = [ cv2.KeyPoint( p.locY,p.locX,size) for p in FeaturesTerminations ] 
    
    kps_bifurc = [ cv2.KeyPoint(p.locY,p.locX,size) for p in FeaturesBifurcations ] 
    
    result=[*kps_term,*kps_bifurc]
    return result





def get_feature_keypoint_and_descriptor(image,orb,padding,border=1):
    DispImg, FeaturesTerminations, FeaturesBifurcations = extract_minutiae_features2(image, showResult=False)
    keypoints=minutiaeToKeyPoints_old( FeaturesTerminations, FeaturesBifurcations)
    _, des = orb.compute(image, keypoints)
    #print('kp',kp)
    #print('des',des)
    return (keypoints, des)
