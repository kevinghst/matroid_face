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