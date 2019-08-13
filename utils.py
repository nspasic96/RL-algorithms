import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, Flatten, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from skimage.io import imshow
from skimage.color import rgb2grey
from skimage.transform import resize
from skimage.viewer import ImageViewer
import os
import tensorflow as tf
import time 
from keras.models import load_model, Model, clone_model
from keras.losses import categorical_crossentropy

def newExploration(minn,maxx,step,STEPS_TO_DECREASE):
    return maxx + step*(minn-maxx)/STEPS_TO_DECREASE

def stateWithHistory(stateBuffer,STATE_BUFFER_SIZE):
    concList = [stateBuffer[i] for i in range(STATE_BUFFER_SIZE-1,-1,-1)]
    return np.concatenate(concList, axis = 2)

def processState(state, stateBuffer, STATE_BUFFER_SIZE):
    state = transformState(state)
    stateBuffer.append(state)
    state = stateWithHistory(stateBuffer, STATE_BUFFER_SIZE)
    return state

def transformState(state):
    shapeToResize = [110, 84]
    grey = rgb2grey(state)
    greyResized = resize(grey, shapeToResize)
    offset = [12,8]
    cropped = greyResized[offset[0]:-offset[1],:]
    final = np.expand_dims(cropped,2)

    #viewer = ImageViewer(np.squeeze(final[:,:],2))
    #viewer.show() 
    return final

def discountAndNormalize(rewards,GAMMA,normalize = False):
    n = len(rewards)

    res = [rewards[n-1]]
    runningAdd = rewards[n-1]
    for i in range(n-2,-1,-1):
        runningAdd*=GAMMA
        runningAdd+=rewards[i]
        aux = [runningAdd]
        aux.extend(res)
        res = aux

    res = np.array(res)
    res = np.expand_dims(res,1)
    #print("res shape is {}".format(res.shape))

    if(normalize):
        res -= np.mean(res)
        res /= np.std(res)

    return res
    
def get_batch_from_memory(idxs, memory, target_network, q_network):

    states=[]
    states_next=[]
    targets =[]
    actions = []
    rewards = []
    dones = []

    for idx in idxs:

        state, action, reward, state_next, done = memory[idx]
        states.append(np.expand_dims(state,0))
        states_next.append(np.expand_dims(state_next,0))
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

    states = np.concatenate(states, axis=0)
    state_next = np.concatenate(states_next, axis=0)

    outputs = target_network.predict(state_next)
    #print("Ouputs dims should be {}x{} and they are {}".format(BATCH_SIZE,NUMBER_OF_ACTIONS,outputs.shape))
    target_f = q_network.predict(states)
    #print("target_f dims should be {}x{} and they are {}".format(BATCH_SIZE,NUMBER_OF_ACTIONS,target_f.shape))

    for i in range(target_f.shape[0]):
        #print(outputs[0].shape)
        target = rewards[i] + (1-dones[i])*GAMMA*np.amax(outputs[i])
        #print("target = {}".format(target))
        target_f[i][actions[i]] = target

    targets = target_f
    #print("states shape {}, targets shape {}".format(states.shape, targets.shape))

    return states, targets
