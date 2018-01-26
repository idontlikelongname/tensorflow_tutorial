import numpy as np
import os
import shutil



def one_hot_encoded(class_numbers, num_classes=None):
    
    if num_classes is None:
        num_classes = np.max(class_numbers)+1
        
    return np.eye(num_classes,dtype=float)[class_numbers]