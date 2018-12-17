from keras.models import Model, load_model, model_from_json
from keras.layers import Activation
from keras.applications.mobilenet import relu6, DepthwiseConv2D

import time
import json
import numpy as np
from glob import glob
import os.path as path
import cv2
def get_data():

     dest_size = (224,224)
  
     data_paths = glob("./test/*.json")
     X=[]
     y=[]
     for data_path in data_paths:
          with open(data_path,'r') as f:
            data = json.loads(f.read())
            y.append(int(data['lightState']))
          img = cv2.imread('{}.jpg'.format(path.splitext(data_path)[0]))
          img = cv2.resize(img,dest_size)
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          img = (img-127.5)/128.0
          X.append(img)
     y=np.array(y)
     X=np.array(X)
     return X,y

def test():
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

     X,y = get_data()

     prediction = model.predict(X)
     max_index = np.argmax(prediction,axis = 1)

     correct = np.where(y==max_index)
     accuracy = np.float32(correct[0].shape[0])/X.shape[0]

     print 'Accuracy is: {}'.format(accuracy)

if __name__=='__main__':
   test()
