# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.datasets import cifar10 # subroutines for fetching the CIFAR-10 dataset
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import os
from scipy import misc

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
arr = misc.imread('celtatest.png')  # 640x480x3 array
arr=arr/255
x=np.array([arr])
y=1
y= np_utils.to_categorical(y, 4) # One-hot encode the labels
print(y)
print(loaded_model.predict(x))
# evaluate loaded model on test data
score = loaded_model.evaluate(x, y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))