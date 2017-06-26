# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.datasets import cifar10 # subroutines for fetching the CIFAR-10 dataset
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import os

(X_train, y_train), (X_test, y_test) = cifar10.load_data() # fetch CIFAR-10 data

num_train, height, width, depth = X_train.shape # there are 50000 training examples in CIFAR-10
num_test = X_test.shape[0] # there are 10000 test examples in CIFAR-10
num_classes = np.unique(y_train).shape[0] # there are 10 image classes

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= np.max(X_train) # Normalise data to [0, 1] range
X_test /= np.max(X_test) # Normalise data to [0, 1] range

filtered_X_train=[]
filtered_y_train=[]
filtered_X_test=[]
filtered_y_test=[]

for pos,i in enumerate(y_train):
    if i==0 or i==1 or i==8 or i==9:
        filtered_X_train.append(X_train[pos])
        filtered_y_train.append(y_train[pos])
for pos, i in enumerate(y_test):
    if i==0 or i==1 or i==8 or i==9:
        filtered_X_test.append(X_test[pos])
        filtered_y_test.append(y_test[pos])

filtered_X_train=np.array(filtered_X_train)
filtered_y_train=np.array(filtered_y_train)
filtered_X_test=np.array(filtered_X_test)
filtered_y_test=np.array(filtered_y_test)

filtered_Y_train = np_utils.to_categorical(filtered_y_train, num_classes)  # One-hot encode the labels
filtered_Y_test = np_utils.to_categorical(filtered_y_test, num_classes)  # One-hot encode the labels

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

print(filtered_X_train[0])
print(filtered_Y_train[0])

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(filtered_X_train, filtered_Y_train, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))