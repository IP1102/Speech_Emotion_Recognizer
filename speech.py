import os
import glob
import matplotlib.pyplot as plt
import librosa
import pandas as pd
import numpy as np

#Load the data
path = 'data/'
lst = []

for subdir, dirs, files in os.walk(path):
  for file in files:
      try:
        #Load librosa array and obtain mfcss
        X, sample_rate = librosa.load(os.path.join(subdir,file), res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
        #TO CONVERT THE LABELS TO 0-7
        file = int(file[7:8]) - 1 
        arr = mfccs, file
        lst.append(arr)
      except ValueError:
        continue

#Converting to X (Dependent) and Y(Independent)
X,y = zip(*lst)
X = np.asarray(X)
Y = np.asarray(y)

#Split into train and test sets (80% - 20%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.20)

#Expand dimension for CNN
X_train = np.expand_dims(X_train, axis = 2)
X_test = np.expand_dims(X_test, axis = 2)

#CNN

import tensorflow as tf
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model

model = Sequential()

#60% acc.
model.add(Conv1D(128,5, padding = 'same',input_shape=(40,1),activation = 'relu'))
model.add(MaxPooling1D(pool_size=(8), padding = 'same'))
model.add(Dropout(0.1))
model.add(Conv1D(128,5, padding = 'same' ,activation = 'relu'))
model.add(MaxPooling1D(pool_size=(8), padding = 'same'))
# model.add(Conv1D(128,5, padding = 'same' ,activation = 'relu'))
# model.add(MaxPooling1D(pool_size=(8), padding = 'same'))
model.add(Dropout(0.1)) #To assign 0 weights to 10% of the neurons
model.add(Flatten())
model.add(Dense(8, activation = 'softmax')) #8 type of emotions

# 98& accuracy, LOL xD Overfit
model.add(Conv1D(100,5, padding = 'same',input_shape=(40,1),activation = 'relu'))
model.add(Dropout(0.1))
model.add(Conv1D(100,5, padding = 'same' ,activation = 'relu'))
model.add(MaxPooling1D(pool_size=(3), padding = 'same'))
model.add(Conv1D(160,5, padding = 'same' ,activation = 'relu'))
model.add(Conv1D(160,5, padding = 'same' ,activation = 'relu'))
model.add(MaxPooling1D(pool_size=(3), padding = 'same'))
model.add(Dropout(0.1)) #To assign 0 weights to 10% of the neurons
model.add(Flatten())
model.add(Dense(8, activation = 'softmax')) #8 type of emotions

#91% - My Best
model.add(Conv1D(120,5, padding = 'same',input_shape=(40,1),activation = 'relu'))
model.add(MaxPooling1D(pool_size=(6), padding = 'same'))
model.add(Conv1D(120,5, padding = 'same' ,activation = 'relu'))
model.add(Dropout(0.15)) #To assign 0 weights to 15% of the neurons
model.add(Flatten())
model.add(Dense(8, activation = 'softmax')) #8 type of emotions





#Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(X_train,y_train,batch_size = 16, epochs = 100, validation_data = (X_test, y_test))
    
model.save('my_model_91.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
