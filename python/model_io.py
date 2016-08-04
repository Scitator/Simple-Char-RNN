'''
Created on 29 July 2016
@author: Kolesnikov Sergey
'''
import numpy as np
import pickle
import os

def save_params(params : dict,
                filepath : str):
    """
    writes the model params to pickled file
    params should look like:
    {
        "weight_matrix_name" : numpy_matrix,
        ...,
        "bias_vector_name" : numpy vector,
        ...
    }
    """
    print ('writing model pickled to {}'.format(os.path.abspath(filepath)))
    with open(filepath, 'wb') as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

def load_params(filepath : str):
    """
    return the model params from pickled file
    params looks like:
    {
        "weight_matrix_name" : numpy_matrix,
        ...,
        "bias_vector_name" : numpy vector,
        ...
    }
    """
    print ('loading pickled model from {}'.format(os.path.abspath(filepath)))
    return pickle.load(open(filepath, 'rb'))
