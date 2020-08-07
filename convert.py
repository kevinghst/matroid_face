import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image

if True:
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), name= 'conv1_1'))
    model.add(Activation('relu', name='relu1_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), name= 'conv1_2'))
    model.add(Activation('relu', name='relu1_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='pool1'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), name= 'conv2_1'))
    model.add(Activation('relu', name='relu2_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), name= 'conv2_2'))
    model.add(Activation('relu', name='relu2_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='pool2'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), name= 'conv3_1'))
    model.add(Activation('relu', name='relu3_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), name= 'conv3_2'))
    model.add(Activation('relu', name='relu3_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), name= 'conv3_3'))
    model.add(Activation('relu', name='relu3_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='pool3'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), name= 'conv4_1'))
    model.add(Activation('relu', name='relu4_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), name= 'conv4_2'))
    model.add(Activation('relu', name='relu4_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), name= 'conv4_3'))
    model.add(Activation('relu', name='relu4_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='pool4'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), name= 'conv5_1'))
    model.add(Activation('relu', name='relu5_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), name= 'conv5_2'))
    model.add(Activation('relu', name='relu5_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), name= 'conv5_3'))
    model.add(Activation('relu', name='relu5_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='pool5'))

    model.add(Convolution2D(4096, (7, 7), name= 'fc6'))
    model.add(Activation('relu', name='relu6'))
    model.add(Dropout(0.5, name='dropout6'))
    model.add(Convolution2D(4096, (1, 1), name= 'fc7'))
    model.add(Activation('relu', name='relu7'))
    model.add(Dropout(0.5, name='dropout7'))
    model.add(Convolution2D(2622, (1, 1), name= 'fc8'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Activation('softmax', name= 'softmax'))

from scipy.io import loadmat

data = loadmat('vgg_face.mat', matlab_compatible=False, struct_as_record=False)

net = data['net'][0][0]

ref_model_layers = net.layers

ref_model_layers = ref_model_layers[0]

for layer in ref_model_layers:
    print(layer[0][0].name)

    
num_of_ref_model_layers = ref_model_layers.shape[0]

base_model_layer_names = [layer.name for layer in model.layers]

for layer in model.layers:
    layer_name = layer.name
    try:
        print(layer_name,": ",layer.weights[0].shape)
    except:
        print("",end='')
        #print(layer_name)

for i in range(num_of_ref_model_layers):
    ref_model_layer = ref_model_layers[i][0,0].name[0]
    
    try:
        weights = ref_model_layers[i][0,0].weights[0,0]
        print(ref_model_layer,": ",weights.shape)
    except:
        #print(ref_model_layer)
        print("",end='')

for i in range(num_of_ref_model_layers):
    ref_model_layer = ref_model_layers[i][0,0].name[0]
    if ref_model_layer in base_model_layer_names:
        #we just need to set convolution and fully connected weights
        if ref_model_layer.find("conv") == 0 or ref_model_layer.find("fc") == 0:
            print(i,". ",ref_model_layer)
            base_model_index = base_model_layer_names.index(ref_model_layer)
            
            weights = ref_model_layers[i][0,0].weights[0,0]
            bias = ref_model_layers[i][0,0].weights[0,1]
            
            model.layers[base_model_index].set_weights([weights, bias[:,0]])

model.save_weights('vgg_face_weights.h5')