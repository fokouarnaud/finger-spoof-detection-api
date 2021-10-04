import cv2
import numpy as np
from .utils2 import *

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
  
 
def get_feature_keypoint_and_descriptor(image,orb,padding,border=1):
    kp, des = orb.detectAndCompute(image, None)
    #print('kp',kp)
    #print('des',des)
    return (kp, des)