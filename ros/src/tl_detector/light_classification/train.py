from glob import glob
import json
import os.path as path
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense,Flatten,Input,Dropout
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Model
BATCH_SIZE = 64

def get_data():

  dest_size = (224,224)

  data_paths = glob("./images/*.json")
  X=[]
  y=[]

  for data_path in data_paths:
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
     label = np.zeros(3)
     label[data['lightState']]=1
     labels.append(label)
  y=np.array(labels)
  X=np.array(X)
  print('Total samples',len(X))
  return X,y

def Network():

  base = MobileNet(pooling = 'avg',weights='imagenet',include_top = True,dropout=.6)
  x = base.output
  x = Dense(1000,name='1000fc',bias_regularizer=tf.keras.regularizers.l2(0.01),activation='relu')(x)
  x = Dropout(0.2,name='Dropout_1')(x)
  x = Dense(3, activation='softmax')(x)
  model = Model(inputs=[base.input],outputs=[x])

  return model




def generator(samples_x,samples_y,batch_size):
  num_samples =  len(samples_x)
  while True:
     shuffle(samples_x,samples_y)
     for offset in range(0,num_samples,batch_size):
        X,y = samples_x[offset:offset+batch_size],samples_y[offset:offset+batch_size]
     yield X,y

def train():
  X,y = get_data()

  train_x,val_x,train_y,val_y = train_test_split(X,y,test_size= 0.2)

  train_steps = len(train_x)//BATCH_SIZE
  val_steps = len(val_x)//BATCH_SIZE

  train_gen = generator(train_x,train_y,BATCH_SIZE)
  val_gen = generator(val_x,val_y,BATCH_SIZE)
  model = Network()
  model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
  model.summary()
  history = model.fit_generator(train_gen,steps_per_epoch=train_steps,epochs=10,verbose=True,validation_data=val_gen,validation_steps=val_steps)

  model_name = './MobileNet_model.h5'
  model.save(model_name)


if __name__=='__main__':
   train()
