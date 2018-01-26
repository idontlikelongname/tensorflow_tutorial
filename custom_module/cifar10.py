# Implemented in Python 3.5

import numpy as np
import pickle
import os
import math

from . import download
from .dataset import one_hot_encoded

# URL for the data-set on the internet.
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
data_path = "/mnt/hdisk/dataset/"

img_size = 32
num_channels = 3
img_size_flat = img_size*img_size*num_channels
num_classes = 10


_num_files_train = 5
_images_per_file = 10000
_num_images_train = _num_files_train*_images_per_file


def _get_file_path(filename=""):
    
    return os.path.join(data_path,"cifar-10-batches-py/",filename)



def _unpickle(filename):
    """
    Unpickle the given file and return the data.
    
    Note that the appropriate dir-name is prepended the filename.
    """
    
    file_path = _get_file_path(filename)
    
    print("Loading data: "+file_path)
    
    with open(file_path, mode='rb') as file:
        # in python 3.x it is important to set the encoding.
        data = pickle.load(file, encoding='bytes')
        
    return data



def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and 
    return a 4-dim array with shape:[image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """
    
    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float)/255.0
    
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    
    images = images.transpose([0,2,3,1])
    
    return images



def _load_data(filename):
    
    data = _unpickle(filename)
    
    raw_images = data[b'data']
    
    # class-numbers for each images. convert to numpy-array.
    cls = np.array(data[b'labels'])
    
    images = _convert_images(raw_images)
    
    return images, cls


################### public method ###############################

def maybe_download_and_extract():
    """
    Download and extract the CIFAR-10 data-set if it doesn't already exist
    in data_path (set this variable first to the desired path).
    """
    download.maybe_download_and_extract(url=data_url, download_dir=data_path)
    print("test")
    

def load_class_names():
    
    # Load the class-names from the pickled file.
    raw = _unpickle(filename="batches.meta")[b'label_names']
    
    # convert from binary strings.
    names = [x.decode('utf-8') for x in raw]
    
    return names


def load_training_data():
    
    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)
    
    begin = 0
    
    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)



def load_test_data():
    """
    Load all the test-data for the CIFAR-10 data-set.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    images, cls = _load_data(filename="test_batch")

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)
    
    
    
def random_mini_batches(X, Y, num_complete_minibatches, mini_batch_size=64, seed=0):
    """
    random shuffer mini batches
    """
    np.random.seed(seed)
    m = X.shape[0]
    mini_batches=[]

    # shuffle
    permutation=list(np.random.permutation(m))
    shuffled_X=X[permutation,...]
    shuffled_Y=Y[permutation,:]

    for k in range(0,num_complete_minibatches):
        mini_batch_X = X[k*mini_batch_size:(k+1)*mini_batch_size,...]
        mini_batch_Y = Y[k*mini_batch_size:(k+1)*mini_batch_size,:]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        end = int(mini_batch_size*num_complete_minibatches)
        mini_batch_X = X[end:m,...]
        mini_batch_Y = Y[end:m,:]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
    
    
    
    
    
    
    