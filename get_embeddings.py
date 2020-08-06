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


def process_folders(root):
    x = []
    y = []
    for folder in os.listdir(root):
        if folder[-1] == 'F':
            gender = 0
        else:
            gender = 1

        folder = root + '/' + folder
        for filename in os.listdir(folder):
            file_path = folder + '/' + filename
            img = load_img(file_path, target_size=(224,224))
            img=img_to_array(img)
            img=np.expand_dims(img,axis=0)
            img=preprocess_input(img)
            img_encode=vgg_face(img)
            embedding = np.squeeze(K.eval(img_encode)).tolist()            
            x.append(embedding)
            y.append(gender)

    return x, y


#print("getting test embeddings...")
#start = time.time()

#root_test = 'data/combined/valid'
#x_test, y_test = process_folders(root_test)

#x_test=np.array(x_test)
#y_test=np.array(y_test)

#with open('x_test.npy', 'wb') as f:
#    np.save(f, x_test)

#with open('y_test.npy', 'wb') as f:
#    np.save(f, y_test)

#end = time.time()
#print("test embeddings took:")
#print(end - start)

print("getting train embeddings...")
start = time.time()

root_train = 'data/combined/aligned'
x_train, y_train = process_folders(root_train)

x_train=np.array(x_train)
y_train=np.array(y_train)

with open('x_train.npy', 'wb') as f:
    np.save(f, x_train)

with open('y_train.npy', 'wb') as f:
    np.save(f, y_train)

end = time.time()
print("train embeddings took:")
print(end - start)



