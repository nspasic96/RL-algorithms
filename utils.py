from matplotlib import pyplot as plt
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
from PIL import Image

import matplotlib.pyplot as plt

def newExploration(minn,maxx,step,STEPS_TO_DECREASE):
    return maxx + step*(minn-maxx)/STEPS_TO_DECREASE

def stateWithHistory(stateBuffer,STATE_BUFFER_SIZE):
    concList = [stateBuffer[i] for i in range(STATE_BUFFER_SIZE-1,-1,-1)]
    return np.concatenate(concList, axis = 2)

def stateWithHistory2(stateBuffer, STATE_BUFFER_SIZE):
    assert(len(stateBuffer) == 2)
    res = stateBuffer[1] - stateBuffer[0]
    res = np.expand_dims(res,2)
    return res

def processState(state, stateBuffer, STATE_BUFFER_SIZE):
    state = transformState(state)
    stateBuffer.append(state)
    state = stateWithHistory(stateBuffer, STATE_BUFFER_SIZE)
    return state

def processState2(state, stateBuffer, STATE_BUFFER_SIZE):
    
    state = transformState2(state)
    stateBuffer.append(state)
    state = stateWithHistory2(stateBuffer, STATE_BUFFER_SIZE)
    return state

def transformState(state):
    shapeToResize = [110, 84]
    grey = rgb2grey(state)
    greyResized = resize(grey, shapeToResize)
    offset = [12,8]
    cropped = greyResized[offset[0]:-offset[1],:]
    final = np.expand_dims(cropped,2)

    final[np.abs(final - 0.32680196) < 1e-3] = 0 # erase background
    final[np.abs(final - 0.9254902) < 1e-3] = 0 # erase background
    final[final != 0] = 1 # set paddles and ball to 1
    
    #viewer = ImageViewer(np.squeeze(final[:,:],2))
    #viewer.show() 
    return final

def transformState2(state):
    shapeToResize = [110, 84]

    #img = Image.fromarray(state, mode="RGB")
    #img.save("slika.png")
    grey = rgb2grey(state)
    greyResized = resize(grey, shapeToResize)
    offset = [12,8]
    final = greyResized[offset[0]:-offset[1],:]
    
    final[np.abs(final - 0.32680196) < 1e-3] = 0 # erase background
    final[np.abs(final - 0.9254902) < 1e-3] = 0 # erase background
    final[final != 0] = 1 # set paddles and ball to 1
    
    #hist = np.histogram(final)
    #print(hist)
    #plt.imshow(final, cmap='gray', vmin=0, vmax=1)
    #plt.show()

    return final

def discountAndNormalize(rewards,GAMMA,normalize = False):

    discounted_r = np.zeros_like(rewards)
    running_add = 0
    for t in range(len(rewards)-1, -1, -1):
        if rewards[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * GAMMA + rewards[t]
        discounted_r[t] = running_add

    res = np.array(discounted_r)
    res = np.expand_dims(res,1)

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
