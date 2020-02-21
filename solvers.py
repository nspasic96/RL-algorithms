import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, Flatten, Activation, BatchNormalization, Dropout
from keras.optimizers import RMSprop, Adam
from skimage.io import imshow
from skimage.color import rgb2grey
from skimage.transform import resize
from skimage.viewer import ImageViewer
import os
import tensorflow as tf
import time 
from keras.models import load_model, Model, clone_model
from keras.losses import categorical_crossentropy, mean_squared_error
import keras.backend as K

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
            
            if(t == "PG"):

                def custom_loss(y_true, y_pred):
                    
                    out = K.clip(y_pred, 1e-8, 1-1e-8)
                    log_lik = y_true*K.log(out)

                    return K.sum(-log_lik*advantages)
            
                input = Input(shape=(self.input_size))
                advantages = Input(shape=[1])
                l = Conv2D(8,4, strides = (2,2), activation = Activation("relu"))(input)
                if(batch_norm):
                    l = BatchNormalization()(l)
                l = Conv2D(16,4, strides = (2,2), activation = Activation("relu"))(l)
                if(batch_norm):
                    l = BatchNormalization()(l)
                l = Flatten()(l)
                l = Dense(20, activation = Activation('relu'))(l)
                if(batch_norm):
                    l = BatchNormalization()(l)
                
                probs = Dense(num_of_act, activation = Activation('softmax'))(l)
                policy = Model(input=[input, advantages], output=[probs])
                policy.compile(optimizer = RMSprop(lr), loss = custom_loss)

                model = Model(input=[input], output=[probs])

            elif(t == "DQN"):

                input = Input(shape=(self.input_size))

                l = Conv2D(8,8, strides = (4,4), activation = Activation("relu"))(input)
                if not test and dropout > 0:
                    l = Dropout(dropout)(l)                           
                if(batch_norm):
                    l = BatchNormalization()(l)
                l = Conv2D(8,4, strides = (2,2), activation = Activation("relu"))(l)
                if not test and dropout > 0:
                    l = Dropout(dropout)(l)                
                if(batch_norm):
                    l = BatchNormalization()(l)
                l = Flatten()(l)
                l = Dense(100, activation = Activation('relu'))(l)
                if(batch_norm):
                    l = BatchNormalization()(l)
                
                qVals = Dense(num_of_act)(l)
                if not test and dropout > 0:
                    l = Dropout(dropout)(l)                
                
                model = Model(input=[input], output=[qVals])
                model.compile(optimizer = Adam(lr), loss = mean_squared_error)
                policy = None
        elif inp == "vector":            
            model = Sequential()
            for i in vector_dims[1:]:
                fc = Dense(i, activation = Activation('relu'))
                model.add(fc)
                if not test and dropout > 0:
                    model.add(Dropout(dropout))
                if(batch_norm):
                    bn = BatchNormalization()
                    model.add(bn)
            
            if(t == "PG"):
                outputs = Dense(num_of_act, activation = Activation('softmax'))
                model.compile(optimizer = Adam(lr), loss =categorical_crossentropy)
            elif(t == "DQN"):
                outputs = Dense(num_of_act)
                model.compile(optimizer = Adam(lr), loss ="mse")
                policy = None
            model.add(outputs)
            
            model.build([None, vector_dims[0]])

        print(model.summary())
        self.policy = policy
        self.model = model
    
    def _get_weights(self):
        return self.model.get_weights()

    def _set_weights(self, new_weights):
        self.model.set_weights(new_weights)

    def fit(self, states, advantages, targets, epochs, batch_size = 32):
        if self.policy is None:
            self.model.fit(states, targets, epochs = epochs, batch_size = batch_size, verbose=0)#this is for DQN
        else:
            self.policy.fit([states, advantages], targets, epochs = epochs, batch_size = batch_size, verbose=0)#this is for PG
    def predict(self, states):
        return self.model.predict(states)

    
    def next_move(self, state, epsilon, doPrint = False):
        rand = False
        next_move = -1
        certainties = -1
        if random.random() < epsilon:
            next_move = np.random.randint(0,self.num_of_act)
            rand = True
        else:
            #certainties = self.predict(np.expand_dims(state,0))
                    
            certainties = self.predict(np.expand_dims(state,0))
            if self.policy is None:

                a = certainties[0,0]
                b = certainties[0,1]
                a = np.exp(a)
                b = np.exp(b)
                s = a+b
                a /= s
                b /= s

                next_move = np.random.choice(np.arange(self.num_of_act), p = [a,1-a])
            else:
                #next_move = np.argmax(certainties[0])
                next_move = np.random.choice(np.arange(self.num_of_act), p = certainties[0])

        if(doPrint):
            print("Next move is chosen. Move random = {}, action selected = {} with actions certainties {}".format(rand, next_move, certainties))
        return next_move

    def load_weights(self, path):
        self.model.load_weights(path)
    def save_weights(self, step, path):
        self.model.save_weights(path + '/model_weights_{}.h5'.format(step))
    def save_weights_for_max(self, e,curr_maxx, path):
        self.model.save_weights(path + '/model_weights_{}_max_{}.h5'.format(e, curr_maxx))

