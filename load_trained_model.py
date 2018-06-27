# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.datasets import cifar10 # subroutines for fetching the CIFAR-10 dataset
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import os


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_weights.h5")
print("Loaded model from disk")

