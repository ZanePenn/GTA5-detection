#Modified code of Sentex Pygta5 2. train_model.py

import numpy as np
import cv2
import time
import os
import pandas as pd
from collections import deque
from random import shuffle
import pickle
from xception import get_model
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras import Input
from keras.callbacks import TensorBoard
from keras.models import load_model

BATCH = 8

balanced_train_path = './balanced_train_data/'

FILE_I_END = -1 #-1 for auto detect #Note: starting file is training_data-0.npy not training_data-1.npy

WIDTH = 480
HEIGHT = 270
EPOCHS = 30 #i am too lazy to put it on while true

MODEL_NAME = 'gta5-{}'.format('xception')

LOAD_MODEL = False

if FILE_I_END == -1:
    FILE_I_END = 1
    while True:
        file_name = balanced_train_path + 'training_data-{}.npy'.format(FILE_I_END)

        if os.path.isfile(file_name):
            print('File exists: ',FILE_I_END)
            FILE_I_END += 1
        else:
            FILE_I_END -= 1
            print('Final File: ',FILE_I_END) 
            break

input_tensor = Input(shape=(HEIGHT,WIDTH,3))
# model = InceptionV3(include_top=True, input_tensor=input_tensor , pooling='max', classes=9, weights=None)
# model.compile('Adagrad', 'categorical_crossentropy')
sess =  tf.compat.v1.Session() 
model = get_model(sess)

"""
tensorboard = TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=True, write_images=True,
    update_freq='epoch', profile_batch=2, embeddings_freq=0,
    embeddings_metadata=None
)
"""

    # if LOAD_MODEL:
    #     model = load_model(MODEL_NAME)
    #     print('We have loaded a previous model!')
        
    # iterates through the training files
for e in range(EPOCHS):
    data_order = [i for i in range(1,FILE_I_END+1)]
    shuffle(data_order)
    for count,i in enumerate(data_order):
        try:
            file_name = balanced_train_path + 'training_data-{}.npy'.format(i)
            # full file info
            train_data = np.load(file_name, allow_pickle=True)

            SAMPLE = len(train_data)
            print('training_data-{}.npy - Sample Size: {} - Batch Size: {}'.format(i,SAMPLE,BATCH))
            
            X = np.array([i[0] for i in train_data])#.reshape(-1,WIDTH,HEIGHT,3) #Pre reshaped at recording
            Y = np.array([i[1] for i in train_data])
            
            print("============================")
            print("Epochs: {} - Steps: {}".format(e, count))
            print(model)
            model.fit(X, Y, batch_size=8 ,epochs=1, validation_split=0.02) #, callbacks=[tensorboard])
    
            print("============================")
            if count%5 == 0 and count != 0:
                print('SAVING MODEL!')
                model.save(MODEL_NAME)
                    
        except Exception as e:
            print(str(e))

print("FINISHED {} EPOCHS!".format(EPOCHS))