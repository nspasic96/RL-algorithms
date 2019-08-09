import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, Flatten, Activation
from keras.optimizers import Adam

ENV_NAME = "SpaceInvaders-v0"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20
APPLY_STEPS = 1

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

INPUT_SIZE = [210,160,3]
NUMBER_OF_ACTIONS = 6

def get_batch_from_memory(idxs, memory, target_network):

    states=[]
    targets =[]
    for idx in idxs:
        state, action, reward, state_next = memory[idx]
        states.append(state)
        outputs = target_network.predict(state_next)
        onehot = np.zeros(shape=(NUMBER_OF_ACTIONS,1))
        onehot[action, 0] = 1
        targets.append(reward + GAMMA*np.amax(outputs)*onehot)

    states = np.array(states)
    states = np.reshape(states, [-1, *INPUT_SIZE])

    targets = np.array(targets)
    targets = np.reshape(targets, [-1, NUMBER_OF_ACTIONS])

    return states, targets


class DQNSolver:
    def __init__(self):

        conv1 = Conv2D(32,3, strides = (4,4))
        conv2 = Conv2D(64,3, strides = (2,2))
        conv3 = Conv2D(64,3, strides = (2,2))
        flat_feature = Flatten()
        fc1 = Dense(50, activation = Activation('relu'))
        outputs = Dense(NUMBER_OF_ACTIONS)

        model = Sequential()
        model.add(conv1)
        model.add(conv2)
        model.add(conv3)
        model.add(flat_feature)
        model.add(fc1)
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
            return np.random.randint(0,6)
        else:
            return np.argmax(self.predict(state))

    

def spaceInvaders(episodes = 1000):
    env = gym.make(ENV_NAME)

    print("Action meanings : {}".format(env.get_action_meanings()))

    q_network = DQNSolver() #sa najsvezijim tezinama
    target_network = DQNSolver()#sa poslednjim zamrznutim tezinama
    memory = deque(maxlen = MEMORY_SIZE)
    
    epsilon = EXPLORATION_MAX

    for e in range(episodes):
        terminal = False
        state = np.expand_dims(env.reset(),0)
        cumulative_reward = 0

        step = 0
        while(not terminal):
            step += 1
            #env.render()
            action = q_network.next_move(state, epsilon)
            state_next, reward, terminal, _ = env.step(action)
            state_next = np.expand_dims(state_next, 0)
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
        
        print("Episode {} over, total reward is {} and exploration rate is now {}".format(e, cumulative_reward, epsilon))

spaceInvaders(1000)