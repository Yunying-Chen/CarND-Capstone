from glob import glob
import json
import os.path as path
import matplotlib.pyplot as plt
import cv2
import keras
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense,Flatten,Input,Dropout
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Model
from keras.callbacks import EarlyStopping

BATCH_SIZE = 4

def get_data():
  return glob("./images/**/*.json")

def get_model():
  base = MobileNet(pooling = 'avg',weights='imagenet',include_top = True,dropout=.6)
  x = base.output
  x = Dense(1000,name='1000fc',bias_regularizer=keras.regularizers.l2(0.01),activation='relu')(x)
  x = Dropout(0.2,name='Dropout_1')(x)
  x = Dense(4, activation='softmax')(x)
  model = Model(inputs=[base.input],outputs=[x])

  return model

def generator(data_paths, batch_size):
  dest_size = (224,224)

  num_samples =  len(data_paths)
  while True:
    
    shuffle(data_paths)
    for offset in range(0, num_samples, batch_size):
      X=[]
      y=[]
      paths = data_paths[offset:offset+batch_size]
      for data_path in paths:
        with open(data_path,'r') as f:
          data = json.loads(f.read())
          y.append(data)

        img = cv2.imread('{}.jpg'.format(path.splitext(data_path)[0]))
        img = cv2.resize(img,dest_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img-127.5)/128.0
        X.append(img)

        h_flip = cv2.flip(img,1)
        X.append(h_flip)
        y.append(data)
    
      labels=[]
      for data in y:
        label = np.zeros(4)
        label[data['lightState']]=1
        labels.append(label)
      y=np.array(labels)
      X=np.array(X)
        
      yield X,y

def train():
  paths = get_data()
  print( 'Total samples: ' + str(len(paths)))


  train_paths, val_paths = train_test_split(paths, test_size= 0.2)

  train_steps = len(train_paths) // BATCH_SIZE
  val_steps = len(val_paths) // BATCH_SIZE

  print( 'val steps ' + str(val_steps))
  print( 'train steps ' + str(train_steps))

  train_gen = generator(train_paths, BATCH_SIZE)
  val_gen = generator(val_paths, BATCH_SIZE)
  model = get_model()
  model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])
  
  # print summary of network architecture
  model.summary()
  early_stopping = EarlyStopping(monitor='val_loss',patience=3)
  history = model.fit_generator(
    train_gen,
    steps_per_epoch=train_steps,  
    epochs=30,verbose=True,
    validation_data=val_gen,
    validation_steps=val_steps,
    callbacks=[early_stopping])

  model_base_name = 'models/mobilenet_sim_model'
  model_name = model_base_name + '.h5'
  model.save(model_name)

  model_json_arch_name = model_base_name + '.json'
  json_arch = model.to_json()
  with open(model_json_arch_name, 'w') as f:
    f.write(json_arch)

  model_weights_name = model_base_name + '.weights'
  model.save_weights(model_weights_name)


if __name__=='__main__':
   train()
