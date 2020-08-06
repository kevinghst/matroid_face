import pdb

import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K


#Define VGG_FACE_MODEL architecture
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

# Load VGG Face model weights
model.load_weights('vgg_face_weights.h5')

# Remove Last Softmax layer and get model upto last flatten layer with outputs 2622 units
vgg_face=Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)

# Construct folder names
folders_train = []
folders_test = []

for i in range(1,71):
    female = str(i) + '_F'
    male = str(i) + '_M'
    folders_train += [female, male]
    folders_test += [female, male]

# Get face embeddings

batch = 10

def process_folders_batch(folders, root):
    x = []
    y = []

    for folder in folders:
        if folder[-1] == 'F':
            gender = 0
        else:
            gender = 1

    folder = root + '/' + folder
    counter = 0
    imgs = None

    for filename in os.listdir(folder):
        file_path = folder + '/' + filename

        img = load_img(file_path, target_size=(224,224))
        img=img_to_array(img)
        img=np.expand_dims(img,axis=0)

        if imgs is not None:
            imgs = np.concatenate((imgs, img), axis=0)
        else:
            imgs = img

        counter += 1
        if counter % batch == 0:
            imgs = preprocess_input(imgs)
            encodings = vgg_face(imgs)
            embeddings = np.squeeze(K.eval(encodings)).tolist()

            x += embeddings
            labels = [gender] * batch
            y += labels

            counter = 0
            imgs = None

    return x, y

folders_train = ['01_F']
root_train = 'data/combined/aligned'

start = time.time()
print("getting embeddings...")

x_train, y_train = process_folders_batch(folders_train, root_train)

end = time.time()
print(end - start)

x_train=np.array(x_train)
y_train=np.array(y_train)

with open('x_train', 'wb') as f:
    np.save(f, x_train)

with open('x_train', 'rb') as f:
    loaded_x_train = np.load(f)

pdb.set_trace()

exit = "exit"