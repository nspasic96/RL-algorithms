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

    if(normalize):
        print(np.histogram(res))

        res -= np.mean(res)
        res /= np.std(res)

    return res
    
def get_batch_from_memory(idxs, memory, target_network, q_network, GAMMA):

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
        #if(done):
            #print(reward)

    states = np.concatenate(states, axis=0)
    #print("Input to nn is of shape {}".format(states.shape))
    state_next = np.concatenate(states_next, axis=0)

    outputs = target_network.predict(state_next)
    target_f = q_network.predict(states)

    for i in range(target_f.shape[0]):
        target = rewards[i] + (1-dones[i])*GAMMA*np.amax(outputs[i])
        #print("Target is {}".format(target))
        target_f[i][actions[i]] = target

    targets = target_f

    return states, targets
