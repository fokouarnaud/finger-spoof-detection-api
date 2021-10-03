from utils import *
from utils2 import *
import fingerprint_feature_extractor

def prepare_dataset(file_names):
    '''
    Coversion to grayscale and enhancement. Split into training and test set.
    :param file_names: All fingerprint images as file names
    :return: train_set, test_set: 2 dictionaries for training and test,
             where the key is the name of the image and the value is the image itself
    '''
    train_set = {}
    test_set = {}
    data = []  # list of tuples
    temp_label = get_image_class(file_names[0])  # sets the image class (101)

    for filename in file_names:
        img = cv2.imread(filename)
        img= skinDetection(img)
        #gray_img = grayscale_image(img)
        padding,img = enhance_image_target(img)
        #img = enhance_image(img)
        img = thinning(img)
        #img = thinning2(img)
        label = get_image_label(filename)
        print('Processing image {} ...  '.format(label))
        if temp_label != get_image_class(filename):
            train, test = split_dataset(data, 0.2)
            train_set.update(train)
            test_set.update(test)
            temp_label = get_image_class(filename)
            data = []
        data.append((label, (img,padding)))

        if filename == file_names[len(file_names) - 1]:
            train, test = split_dataset(data, 0.2)
            train_set.update(train)
            test_set.update(test)

    print('DONE')
    return train_set, test_set

def prepare_dataset_probe(file_names):
    data = []  # list of tuples
    test_set = {}
    temp_label = get_image_class(file_names[0])  # sets the image class (101)
    print(temp_label)

    for filename in file_names:
        img = cv2.imread(filename)
        img= skinDetection(img)
        #gray_img = grayscale_image(img)
        padding,img = enhance_image_target(img)
        #img = enhance_image(img)
        img = thinning(img)
        #img = thinning2(img)
        label = get_image_label(filename)
        print('Processing image {} ...  '.format(label))  
        data.append((label, (img,padding)))

    test_set.update(data)
    print('DONE test_set')
    return test_set

def prepare_dataset_gallery(file_names):
    
    data = []  # list of tuples
    train_set = {}
    temp_label = get_image_class(file_names[0])  # sets the image class (101)
    print(temp_label)

    for filename in file_names:
        img = cv2.imread(filename)
        img= skinDetection(img)
        #gray_img = grayscale_image(img)
        padding,img = enhance_image_target(img)
        #img = enhance_image(img)
        img = thinning(img)
        #img = thinning2(img)
        label = get_image_label(filename)
        print('Processing image {} ...  '.format(label))
        
        data.append((label, (img,padding)))

    train_set.update(data)
    print('DONE train_set')
    return train_set

def prepare_dataset_authentication(train_feature_descriptors):
    '''
    Splits dataset  by each person data
    for the authentication scenario
    :param train_feature_descriptors: training set
    :return: dictionary where the key denotes the name of the person,
    and the value is a list with all trained feature descriptors
    '''
    authentication_databases = {}
    temp_list = {}
    class_name = get_image_class(list(train_feature_descriptors.keys())[0])  # inital class name
    last_key = list(train_feature_descriptors.keys())[-1]

    for image_id, feature_descriptor in train_feature_descriptors.items():
        if class_name != get_image_class(image_id):
            authentication_databases[class_name] = temp_list
            temp_list = {}
        temp_list[image_id] = feature_descriptor
        class_name = get_image_class(image_id)

        if last_key == image_id:
            authentication_databases[class_name] = temp_list

    return authentication_databases

