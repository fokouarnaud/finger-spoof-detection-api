import cv2
import glob
import numpy as np
import fprmodules.enhancement.image_enhance as fe
import ftlib as ft
from utils2 import *

from sklearn.model_selection import train_test_split
from constants import IMAGES_PATH, IMAGES_PATH_GALLERY, IMAGES_PATH_PROBE


def read_images():
    '''
    Reads all images from IMAGES_PATH and sorts them
    :return: sorted file names
    '''
    file_names = [img for img in glob.glob(IMAGES_PATH + "/*.*")]
    file_names.sort()
    return file_names

def read_images_gallery():
    '''
    Reads all images from IMAGES_PATH and sorts them
    :return: sorted file names
    '''
    file_names = [img for img in glob.glob(IMAGES_PATH_GALLERY + "/*.*")]
    file_names.sort()
    return file_names


def read_images_probe():
    '''
    Reads all images from IMAGES_PATH and sorts them
    :return: sorted file names
    '''
    file_names = [img for img in glob.glob(IMAGES_PATH_PROBE + "/*.*")]
    file_names.sort()
    return file_names

def get_image_label(filename):
    image = filename.split('/')
    return image[len(image)-1]


def get_image_class(filename):
    x = get_image_label(filename).split('_')
    return "{}{}".format(x[0], x[2])

def get_one_image_class(filenames):
    return ''.join(filenames[:-1]);

# Splits the dataset on training and testing set
def split_dataset(data, test_size):
    train, test = train_test_split(data, test_size=test_size, random_state=42)
    return train, test


def grayscale_image(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


# Enhancement using orientation/frequency filtering - Gabor filterbank
def enhance_image(image):
    #img_e, mask1, orientim1, freqim1 = fe.image_enhance(image)
    
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #img = img[250:950, 900:1912]

    rows,cols = np.shape(image)
    aspect_ratio = np.double(rows)/np.double(cols)

    new_rows = 350             # randomly selected number
    new_cols = new_rows/aspect_ratio

    image = cv2.resize(image,(np.int(new_cols),np.int(new_rows)));
    
    img_e = fe.image_enhance(image)
    return cv2.normalize(np.uint8(img_e), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=0)
    #return img_e


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
  
    
def get_genuine_impostor_scores(all_scores, identical):
    '''
    Returns two arrays with the genuine and impostor scores.
    The genuine match scores are obtained by matching feature sets
    of the same class (same person) and the impostor match scores are obtained
    by matching feature sets of different classes (different persons)
    '''
    genuine_scores = []
    impostor_scores = []
    for i in range(0, len(all_scores)):
        if identical[i] == 1:
            genuine_scores.append(all_scores[i][1])
        else:
            impostor_scores.append(all_scores[i][1])

    return genuine_scores, impostor_scores




