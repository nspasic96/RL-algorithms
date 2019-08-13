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
from utils import *
from solvers import Solver
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--ENV_NAME', default = "SpaceInvaders-v0")
parser.add_argument('--inp', default = "vector")
parser.add_argument('--BATCH_SIZE', default = 32)
parser.add_argument('--LEARNING_RATE', default = 0.0003)
parser.add_argument('--EXPLORATION_MAX', default = 1)
parser.add_argument('--EXPLORATION_MIN', default = 0.1)
parser.add_argument('--STEPS_TO_DECREASE', default = 1000000)
parser.add_argument('--SAVE_STEPS', default = 50000)
parser.add_argument('--GAMMA', default = 0.95)
parser.add_argument('--MEMORY_SIZE', default = 1000000)
parser.add_argument('--APPLY_STEPS', default = 11000)
parser.add_argument('--EPISODES_UPDATE', default = 10)
parser.add_argument('--STATE_BUFFER_SIZE', default = 4)
parser.add_argument('--BATCH_NORM', default = 0)
parser.add_argument('--NUMBER_OF_ACTIONS', default = 6)
parser.add_argument('--RUNNING_MEAN_ACC', default = 50)
parser.add_argument('--LOAD_PRETRAINED', default = None)
parser.add_argument('--NORMALIZE_REWARDS', default = 0)

args = parser.parse_args()

ENV_NAME = args.ENV_NAME

BATCH_SIZE = args.BATCH_SIZE
print(type(BATCH_SIZE))
LEARNING_RATE = args.LEARNING_RATE
EXPLORATION_MAX = args.EXPLORATION_MAX
EXPLORATION_MIN = args.EXPLORATION_MIN
STEPS_TO_DECREASE = args.STEPS_TO_DECREASE
SAVE_STEPS = args.SAVE_STEPS
NORMALIZE_REWARDS = bool(int(args.NORMALIZE_REWARDS))

GAMMA = args.GAMMA
MEMORY_SIZE = args.MEMORY_SIZE
APPLY_STEPS = args.APPLY_STEPS
EPISODES_UPDATE = args.EPISODES_UPDATE

inp = args.inp
STATE_BUFFER_SIZE = args.STATE_BUFFER_SIZE

BATCH_NORM = bool(int(args.BATCH_NORM))
NUMBER_OF_ACTIONS = args.NUMBER_OF_ACTIONS
RUNNING_MEAN_ACC = args.RUNNING_MEAN_ACC

LOAD_PRETRAINED = args.LOAD_PRETRAINED
#LOAD_PRETRAINED = r"C:\SpaceInvadors\env_SpaceInvaders-v0_dqn_batch_size=32_apply_steps=11000_state_bufS=4_batch_norm_False_numberAc_6\model_weights_150000.h5"

if inp == "picture":
    INPUT_SIZE = [90,84,STATE_BUFFER_SIZE]
    LAYERS = None
elif inp == "vector":
    STATES_DESC = 4
    if STATE_BUFFER_SIZE == 1:
        INPUT_SIZE = [STATES_DESC]  
    else:
        INPUT_SIZE = [STATES_DESC, STATE_BUFFER_SIZE]
    LAYERS = [STATES_DESC, 50, 20]

def writeParams(path):
    with open(path, "a") as f:
        f.write("GAMMA : {}\n".format(GAMMA))
        f.write("LEARNING_RATE : {}\n".format(LEARNING_RATE))
        f.write("MEMORY_SIZE : {}\n".format(MEMORY_SIZE))
        f.write("BATCH_SIZE : {}\n".format(BATCH_SIZE))
        f.write("APPLY_STEPS : {}\n".format(APPLY_STEPS))
        f.write("EXPLORATION_MAX : {}\n".format(EXPLORATION_MAX))
        f.write("EXPLORATION_MIN : {}\n".format(EXPLORATION_MIN))
        f.write("STEPS_TO_DECREASE : {}\n".format(STEPS_TO_DECREASE))
        f.write("SAVE_STEPS : {}\n".format(SAVE_STEPS))
        f.write("STATE_BUFFER_SIZE : {}\n".format(STATE_BUFFER_SIZE))
        f.write("INPUT_SIZE : {}\n".format(INPUT_SIZE))
        f.write("BATCH_NORM : {}\n".format(BATCH_NORM))

def game(path):
    env = gym.make(ENV_NAME)
    gameScores = []

    #Tensorflow initializations
    init = tf.global_variables_initializer()
    s = tf.placeholder(dtype = tf.float32)
    rew = tf.placeholder(dtype = tf.float32)
    gameScoreMean = tf.reduce_mean(s)
    tf.summary.scalar(tensor = gameScoreMean, name = "mean_last_{}_episodes".format(RUNNING_MEAN_ACC))
    tf.summary.scalar(tensor = rew, name = "reward")
    
    summaries = tf.summary.merge_all()

    pgs = Solver("PG", NUMBER_OF_ACTIONS, LEARNING_RATE,BATCH_NORM,STATE_BUFFER_SIZE,inp=inp, vector_dims=LAYERS)

    cTargets = []
    cInputs = []
    
    epsilon = EXPLORATION_MAX
    
    curr_maxx = -1000
    e=0
    with tf.Session() as sess:

        sess.run(init)
        writer = tf.summary.FileWriter(path + "/train", sess.graph_def)
        step = 0
        while True:
            #Episod starts here
            e+=1

            rewards = []
            actions = []
            memory = []

            stateBuffer = deque(maxlen = STATE_BUFFER_SIZE)
            for _ in range(STATE_BUFFER_SIZE):
                stateBuffer.append(np.zeros(shape=[*INPUT_SIZE[0:-1],1]))

            terminal = False
            state = env.reset()

            if(inp == "picture"):
                state = processState(state, stateBuffer, STATE_BUFFER_SIZE)

            cumulative_reward = 0
            max_reached = 0

            start = time.time()
            while(not terminal):
                step += 1
                #if(step > STEPS_TO_DECREASE):
                env.render()
                action = pgs.next_move(state, epsilon)
                state_next, reward, terminal, _ = env.step(action)
                rewards.append(reward)
                actions.append(action)

                if(inp == "picture"):
                    state_next = processState(state_next, stateBuffer, STATE_BUFFER_SIZE)
                
                memory.append(state)

                cumulative_reward += reward
                state = state_next

                epsilon = newExploration(EXPLORATION_MIN, EXPLORATION_MAX, step, STEPS_TO_DECREASE)
                epsilon = np.clip(epsilon, EXPLORATION_MIN, EXPLORATION_MAX)
                
                if step % SAVE_STEPS == 0:
                    st = time.time()
                    pgs.save_weights(step, path)
                    el = time.time() - st
                    print("{}. steps done in episode {}, work saved in {}s".format(step,e,el))

            discRewards = discountAndNormalize(rewards,GAMMA,NORMALIZE_REWARDS)
            actionsOneHot = np.zeros(shape=(len(actions),NUMBER_OF_ACTIONS))
            for idx,i in enumerate(actions):
                actionsOneHot[idx,i] = 1
            
            targets = discRewards*actionsOneHot
            inputs = np.reshape(memory, [-1, *INPUT_SIZE])

            if(e % EPISODES_UPDATE == 0):
                ttInputs = cInputs[0]
                ttTargets = cTargets[0]
                n = len(cInputs)
                for i in range(1,n):
                    ttInputs = np.vstack((ttInputs, cInputs[i]))
                    ttTargets = np.vstack((ttTargets, cTargets[i]))

                #print("ttInputs shape = {}".format(ttInputs.shape))
                #print("ttTargets shape = {}".format(ttTargets.shape))
                pgs.model.fit(ttInputs, ttTargets, epochs=1, batch_size=128)
                cInputs = []
                cTargets = []
            else:
                cInputs.append(inputs)
                cTargets.append(targets)                

            elapsed = time.time() - start
            gameScores.append(cumulative_reward)

            if(cumulative_reward > curr_maxx):
                curr_maxx = cumulative_reward
                max_reached += 1
                st = time.time()
                pgs.save_weights_for_max(e,curr_maxx,path)
                el = time.time() - st
                print("New max({}) reached! Weights saved in {}s".format(curr_maxx, el))
            
            start = 0
            if(e >= RUNNING_MEAN_ACC):
                start = -RUNNING_MEAN_ACC-1
            
            summary = sess.run(summaries, feed_dict={s : gameScores[start:-1], rew : cumulative_reward})
            writer.add_summary(summary, e)
                
            print("Episode {} over(total {} steps until now) in {}s, total reward is {} and exploration rate is now {}. \n Mean score is {}.".format(e,step, 
                elapsed, cumulative_reward, epsilon, np.mean(gameScores)))

if __name__ == "__main__":

    path = "./env_{}_pg_batch_size={}_apply_steps={}_state_bufS={}_batch_norm_{}_numberAc_{}".format(ENV_NAME,BATCH_SIZE,APPLY_STEPS,STATE_BUFFER_SIZE,BATCH_NORM,NUMBER_OF_ACTIONS)
    if not os.path.exists(path):
        print("Creating folder")
        os.mkdir(path)
    writeParams(path + "/params.txt")
    game(path)
