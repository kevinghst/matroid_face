import pdb

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.math import confusion_matrix

# Load saved data
x_train=np.load('x_train.npy')
y_train=np.load('y_train.npy')
x_test=np.load('x_test.npy')
y_test=np.load('y_test.npy')

# Softmax regressor to classify images based on encoding 
classifier_model=Sequential()
classifier_model.add(Dense(units=100,input_dim=x_train.shape[1],kernel_initializer='glorot_uniform'))
classifier_model.add(BatchNormalization())
classifier_model.add(Activation('tanh'))
classifier_model.add(Dropout(0.3))
classifier_model.add(Dense(units=10,kernel_initializer='glorot_uniform'))
classifier_model.add(BatchNormalization())
classifier_model.add(Activation('tanh'))
classifier_model.add(Dropout(0.2))
classifier_model.add(Dense(1, activation='sigmoid'))
classifier_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer='nadam',metrics=['accuracy'])

# Train model
classifier_model.fit(x_train,y_train,epochs=5,validation_data=(x_test,y_test))

# Report final accuracies
predictions = classifier_model.predict(x_test)
y_pred = np.squeeze(predictions)
y_pred = (y_pred > 0.5)

matrix = tf.math.confusion_matrix(y_pred, y_test)

female_accuracy = float(matrix[0][0] / (matrix[0][0] + matrix[0][1]))
male_accuracy = float(matrix[1][1] / (matrix[1][0] + matrix[1][1]))
overall_accuracy = float((matrix[0][0] + matrix[1][1]) / len(y_test))

print("Female accuracy: %s" % female_accuracy)
print("Male accuracy: %s" % male_accuracy)
print("Overall accuracy: %s" % overall_accuracy)