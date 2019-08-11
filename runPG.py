import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, Flatten, Activation, BatchNormalization
from keras.optimizers import Adam, SGD
from skimage.io import imshow
from skimage.color import rgb2grey
from skimage.transform import resize
from skimage.viewer import ImageViewer
import os
import tensorflow as tf
import time 
from keras.models import load_model, Model, clone_model
from keras.losses import categorical_crossentropy

ENV_NAME = "SpaceInvaders-v0"

GAMMA = 0.99
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 32
APPLY_STEPS = 11000

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
STEPS_TO_DECREASE = 200000

SAVE_STEPS = 50000

STATE_BUFFER_SIZE = 4
INPUT_SIZE = [90,84,STATE_BUFFER_SIZE]
NUMBER_OF_ACTIONS = 6
GOOD_GAME_TH = 100
BATCH_NORM = False
LOAD_PRETRAINED = None
#LOAD_PRETRAINED = r"C:\SpaceInvadors\batch_size=32_apply_steps=10000_state_bufS=4_batch_norm_False_numberAc_6\model_weights_102000.h5"

def newExploration(minn,maxx,step):
    return maxx + step*(minn-maxx)/STEPS_TO_DECREASE

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

class PGSolver:
    def __init__(self):
        #input to network will be (None, 90,84,4)
        conv1 = Conv2D(16,8, strides = (4,4), activation = Activation("relu"))
        bn1 = BatchNormalization()
        conv2 = Conv2D(32,4, strides = (2,2), activation = Activation("relu"))
        bn2 = BatchNormalization()
        flat_feature = Flatten()
        fc1 = Dense(256, activation = Activation('relu'))
        bn3 = BatchNormalization()
        outputs = Dense(NUMBER_OF_ACTIONS, activation = Activation('softmax'))

        model = Sequential()
        model.add(conv1)
        if(BATCH_NORM):
            model.add(bn1)
        model.add(conv2)
        if(BATCH_NORM):
            model.add(bn2)
        model.add(flat_feature)
        model.add(fc1)
        if(BATCH_NORM):
            model.add(bn3)
        model.add(outputs)

        model.compile(optimizer = SGD(LEARNING_RATE), loss = categorical_crossentropy)
        model.build([None, *INPUT_SIZE])
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
        if random.random() < epsilon:
            return np.random.randint(0,NUMBER_OF_ACTIONS)
        else:
            return np.argmax(self.predict(np.expand_dims(state,0)))
    def load_weights(self, path):
        self.model.load_weights(path)
    def save_weights(self, step):
        self.model.save_weights(path + '/model_weights_{}.h5'.format(step))
    def save_weights_for_max(self, e,curr_maxx):
        self.model.save_weights(path + '/model_weights_{}_max_{}.h5'.format(e, curr_maxx))


def processState(state, stateBuffer):
    state = transformState(state)
    stateBuffer.append(state)
    state = stateWithHistory(stateBuffer)
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

def stateWithHistory(stateBuffer):
    concList = [stateBuffer[i] for i in range(STATE_BUFFER_SIZE-1,-1,-1)]
    return np.concatenate(concList, axis = 2)

def discountAndNormalize(rewards):
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

    res -= np.mean(res)
    res /= np.std(res)

    return res

def spaceInvaders():
    print("Trainininininng")
    env = gym.make(ENV_NAME)
    gameScores = []
    goodGameScores = []

    #Tensorflow initializations
    init = tf.global_variables_initializer()
    s = tf.placeholder(dtype = tf.float32)
    rew = tf.placeholder(dtype = tf.float32)
    gameScoreMean = tf.reduce_mean(s)
    mean_summary = tf.summary.scalar(tensor = gameScoreMean, name = "mean")
    res_summary = tf.summary.scalar(tensor = rew, name = "rew")
    
    summaries = tf.summary.merge_all()

    pgs = PGSolver()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter(path + "/train", sess.graph_def)
    
    epsilon = EXPLORATION_MAX
    
    curr_maxx = -1000
    e=0
    with tf.Session() as sess:
        step = 0
        while True:
            e+=1

            rewards = []
            actions = []
            memory = []

            stateBuffer = deque(maxlen = STATE_BUFFER_SIZE)
            for _ in range(STATE_BUFFER_SIZE):
                stateBuffer.append(np.zeros(shape=[*INPUT_SIZE[0:2],1]))

            terminal = False
            state = env.reset()

            state = processState(state, stateBuffer)

            cumulative_reward = 0
            max_reached = 0

            start = time.time()
            while(not terminal):
                step += 1
                env.render()
                action = pgs.next_move(state, epsilon)
                state_next, reward, terminal, _ = env.step(action)
                rewards.append(reward)
                actions.append(action)

                state_next = processState(state_next, stateBuffer)
                memory.append(state)

                cumulative_reward += reward
                state = state_next

                epsilon = newExploration(EXPLORATION_MIN, EXPLORATION_MAX, step)
                epsilon = np.clip(epsilon, EXPLORATION_MIN, EXPLORATION_MAX)
                
                if step % SAVE_STEPS == 0:
                    st = time.time()
                    pgs.save_weights(step)
                    el = time.time() - st
                    print("{}. steps done in episode {}, work saved in {}s".format(step,e,el))

            discRewards = discountAndNormalize(rewards)
            actionsOneHot = np.zeros(shape=(len(actions),NUMBER_OF_ACTIONS))
            for idx,i in enumerate(actions):
                actionsOneHot[idx,i] = 1
            
            targets = discRewards*actionsOneHot
            inputs = np.reshape(memory, [-1, *INPUT_SIZE])
            pgs.fit(inputs, targets, epochs=1)

            elapsed = time.time() - start
            if(cumulative_reward > GOOD_GAME_TH):
                goodGameScores.append(cumulative_reward)
            gameScores.append(cumulative_reward)

            if(cumulative_reward > curr_maxx):
                curr_maxx = cumulative_reward
                max_reached += 1
                st = time.time()
                pgs.save_weights_for_max(e,curr_maxx)
                el = time.time() - st
                print("New max({}) reached! Weights saved in {}s".format(curr_maxx, el))

            summary = sess.run(summaries, feed_dict={s : gameScores, rew : cumulative_reward})
            writer.add_summary(summary, e)
                
            print("Episode {} over(total {} steps until now) in {}s, total reward is {} and exploration rate is now {}. \n Mean score is {}, and filtered mean score is {}(total {} games count)".format(e,step, 
                elapsed, cumulative_reward, epsilon, np.mean(gameScores), np.mean(goodGameScores), len(goodGameScores)))

if __name__ == "__main__":

    path = "./env_{}_pg_batch_size={}_apply_steps={}_state_bufS={}_batch_norm_{}_numberAc_{}".format(ENV_NAME,BATCH_SIZE,APPLY_STEPS,STATE_BUFFER_SIZE,BATCH_NORM,NUMBER_OF_ACTIONS)
    if not os.path.exists(path):
        print("Creating folder")
        os.mkdir(path)
    writeParams(path + "/params.txt")
    spaceInvaders()
