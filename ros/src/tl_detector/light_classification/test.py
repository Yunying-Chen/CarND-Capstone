from keras.models import Model, load_model, model_from_json
from keras.layers import Activation
from keras.applications.mobilenet import relu6, DepthwiseConv2D

import time
import json

model_path = 'models/mobilenet_sim_model.h5'

t = time.time()
model = load_model(model_path, custom_objects={'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D})  
print 'load time of H5 model: {}'.format(time.time() - t)

t = time.time()
with open('models/mobilenet_sim_model.json', 'r') as f:
    json_arch = f.read()
model = model_from_json(json_arch, custom_objects={'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D})
model.load_weights('models/mobilenet_sim_model.weights')
print 'load time of json arch model: {}'.format(time.time() - t) 
