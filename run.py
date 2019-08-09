import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, Flatten, Activation, BatchNormalization
from keras.optimizers import Adam
from skimage.io import imshow
from skimage.color import rgb2grey
from skimage.transform import resize
from skimage.viewer import ImageViewer

ENV_NAME = "SpaceInvaders-v0"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
<<<<<<< Updated upstream
BATCH_SIZE = 64
=======
BATCH_SIZE = 20
>>>>>>> Stashed changes
APPLY_STEPS = 50

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
SAVE_STEPS = 2000

STATE_BUFFER_SIZE = 4
INPUT_SIZE = [90,84,STATE_BUFFER_SIZE]
NUMBER_OF_ACTIONS = 6

def get_batch_from_memory(idxs, memory, target_network):

    states=[]
    targets =[]
    for idx in idxs:
        state, action, reward, state_next = memory[idx]
        states.append(state)
        outputs = target_network.predict(np.expand_dims(state_next,0))
        onehot = np.zeros(shape=(NUMBER_OF_ACTIONS,1))
        onehot[action, 0] = 1
        targets.append(reward + GAMMA*np.amax(outputs)*onehot)

    states = np.array(states)
    states = np.reshape(states, [-1, *INPUT_SIZE])

    targets = np.array(targets)
    targets = np.reshape(targets, [-1, NUMBER_OF_ACTIONS])

    return states, targets


class DQNSolver:
    def __init__(self, batch_norm = False):
        #input to network will be (None, 84,84,4)
        conv1 = Conv2D(16,8, strides = (4,4), activation = Activation("relu"))
        bn1 = BatchNormalization()
        conv2 = Conv2D(32,4, strides = (2,2), activation = Activation("relu"))
        bn2 = BatchNormalization()
        flat_feature = Flatten()
        fc1 = Dense(256, activation = Activation('relu'))
        bn3 = BatchNormalization()
        outputs = Dense(NUMBER_OF_ACTIONS)

        model = Sequential()
        model.add(conv1)
        if(batch_norm):
            model.add(bn1)
        model.add(conv2)
        if(batch_norm):
            model.add(bn2)
        model.add(flat_feature)
        model.add(fc1)
        if(batch_norm):
            model.add(bn3)
        model.add(outputs)

        model.compile(optimizer = Adam(LEARNING_RATE), loss ="mse")
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
            return np.argmax(self.predict(state))

    def save_weights(self, step):
        model.save_weights('model_weights_{}.h5'.format(step))

def processState(state, stateBuffer):
    state = transformState(state)
    stateBuffer.append(state)
    state = stateWithHistory(stateBuffer)
    return state

def transformState(state):
    shapeToResize = [110, 84]
    grey = rgb2grey(state)
    greyResized = resize(grey, shapeToResize)
    #offset = (shapeToResize[0] - shapeToResize[1]) // 2
    offset = [12,8]
    cropped = greyResized[offset[0]:-offset[1],:]
    final = np.expand_dims(cropped,2)
    
    #viewer = ImageViewer(final[:,:,0])
    #viewer.show()
    return final

def stateWithHistory(stateBuffer):
    concList = [stateBuffer[i] for i in range(STATE_BUFFER_SIZE-1,-1,-1)]
    print("concList shape is {}".format(len(concList)))
    return np.concatenate(concList, axis = 2)


def spaceInvaders(episodes = 1000):
    env = gym.make(ENV_NAME)

    print("Action meanings : {}".format(env.get_action_meanings()))

    q_network = DQNSolver() #sa najsvezijim tezinama
    target_network = DQNSolver()#sa poslednjim zamrznutim tezinama
    memory = deque(maxlen = MEMORY_SIZE)
    
    epsilon = EXPLORATION_MAX

    step = 0
    for e in range(episodes):
        stateBuffer = deque(maxlen = STATE_BUFFER_SIZE)
        for _ in range(STATE_BUFFER_SIZE):
            stateBuffer.append(np.zeros(shape=[*INPUT_SIZE[0:2],1]))

        terminal = False
        state = env.reset()

        state = processState(state, stateBuffer)

        cumulative_reward = 0

        while(not terminal):
            step += 1
            print("Step = {}\n".format(step))
            env.render()
            action = q_network.next_move(state, epsilon)
            state_next, reward, terminal, _ = env.step(action)

            state_next = processState(state_next, stateBuffer)
            memory.append((state, action, reward, state_next))

            if(len(memory) >= BATCH_SIZE):
                idxs = np.random.choice(len(memory), BATCH_SIZE, replace=False)
                states, targets = get_batch_from_memory(idxs, memory, target_network)
                q_network.fit(states, targets, epochs = 1)

                if step % APPLY_STEPS == 0:
                    target_network._set_weights(q_network._get_weights())

            cumulative_reward += reward
            state = state_next

            epsilon *= EXPLORATION_DECAY
            epsilon = np.clip(epsilon, EXPLORATION_MIN, EXPLORATION_MAX)

            if step % SAVE_STEPS == 0:
                q_network.save_weights(step)
        
        print("Episode {} over, total reward is {} and exploration rate is now {}".format(e, cumulative_reward, epsilon))

spaceInvaders(1000)