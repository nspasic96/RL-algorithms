import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, Flatten, Activation, BatchNormalization, Dropout
from keras.optimizers import RMSprop
from skimage.io import imshow
from skimage.color import rgb2grey
from skimage.transform import resize
from skimage.viewer import ImageViewer
import os
import tensorflow as tf
import time 
from keras.models import load_model, Model, clone_model
from keras.losses import categorical_crossentropy


class Solver:
    #t : PG or DQN
    #num_of_act : number of possible actions
    #lr : learning rate
    #test : whether solver is used for testing or training
    #dropout : dropout if test is False
    #inp : whether input is an picture or vector
    #vector_dims : list of network layers size(inluding input and exluding output)
    def __init__(self, t, num_of_act, lr, batch_norm, history_length, test=False, dropout=0, inp="picture", vector_dims=None):
        self.num_of_act = num_of_act
        if(history_length > 2):
            self.input_size = [90,84,history_length]
        else:
            self.input_size = [90,84,1] #DIFFERENCE IMAGE
        #input to network will be (None, 90,84,4)
        if inp == "picture":
            conv1 = Conv2D(16,8, strides = (4,4), activation = Activation("relu"))
            bn1 = BatchNormalization()
            conv2 = Conv2D(32,4, strides = (2,2), activation = Activation("relu"))
            bn2 = BatchNormalization()
            flat_feature = Flatten()
            fc1 = Dense(256, activation = Activation('relu'))
            bn3 = BatchNormalization()
            if(t == "PG"):
                outputs = Dense(num_of_act, activation = Activation('softmax'))
            elif(t == "DQN"):
                outputs = Dense(num_of_act)

            model = Sequential()
            model.add(conv1)
            if not test and dropout > 0:
                model.add(Dropout(dropout))
            if(batch_norm):
                model.add(bn1)
            model.add(conv2)
            if not test and dropout > 0:
                model.add(Dropout(dropout))
            if(batch_norm):
                model.add(bn2)
            model.add(flat_feature)
            model.add(fc1)
            if not test and dropout > 0:
                model.add(Dropout(dropout))
            if(batch_norm):
                model.add(bn3)
            model.add(outputs)

            if(t == "PG"):
                model.compile(optimizer = RMSprop(lr), loss = categorical_crossentropy)
            elif(t == "DQN"):
                model.compile(optimizer = RMSprop(lr), loss ="mse")
            model.build([None, *self.input_size])
        elif inp == "vector":            
            model = Sequential()
            for i in vector_dims[1:]:
                fc = Dense(i, activation = Activation('relu'), kernel_initializer='random_normal')
                model.add(fc)
                if not test and dropout > 0:
                    model.add(Dropout(dropout))
                if(batch_norm):
                    bn = BatchNormalization()
                    model.add(bn)
            
            if(t == "PG"):
                outputs = Dense(num_of_act, activation = Activation('softmax'))
            elif(t == "DQN"):
                outputs = Dense(num_of_act)
            model.add(outputs)

            if(t == "PG"):
                model.compile(optimizer = RMSprop(lr), loss = categorical_crossentropy)
            elif(t == "DQN"):
                model.compile(optimizer = RMSprop(lr), loss ="mse")
            
            model.build([None, vector_dims[0]])

        print(model.summary())
        self.model = model
    
    def _get_weights(self):
        return self.model.get_weights()

    def _set_weights(self, new_weights):
        self.model.set_weights(new_weights)

    def fit(self, states, targets, epochs):
        self.model.fit(states, targets, epochs = epochs, verbose=0)

    def predict(self, states):
        return self.model.predict(states)

    def next_move(self, state, epsilon):
        rand = False
        next_move = -1
        certainties = -1
        if random.random() < epsilon:
            next_move = np.random.randint(0,self.num_of_act)
            rand = True
        else:
            #certainties = self.predict(np.expand_dims(state,0))
                    
            certainties = self.predict(np.expand_dims(state,0))
            next_move = np.argmax(certainties)

        #print("Next move is chosen. Move random = {}, action selected = {} with actions certainties {}".format(rand, next_move, certainties), end ='/r')
        return next_move

    def load_weights(self, path):
        self.model.load_weights(path)
    def save_weights(self, step, path):
        self.model.save_weights(path + '/model_weights_{}.h5'.format(step))
    def save_weights_for_max(self, e,curr_maxx, path):
        self.model.save_weights(path + '/model_weights_{}_max_{}.h5'.format(e, curr_maxx))

