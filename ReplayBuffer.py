from collections import deque
import numpy as np

class ReplayBuffer:

    def __init__(self, maxSize):
        self.maxSize = maxSize
        self.obsDeq = deque(maxlen=maxSize)
        self.actionsDeq = deque(maxlen=maxSize)
        self.rewardsDeq = deque(maxlen=maxSize)
        self.nextObssDeq = deque(maxlen=maxSize)
        self.terminalsDeq = deque(maxlen=maxSize)
        self.curr = 0


    def add(self, obs, action, reward, nextObs, terminal):
        self.obsDeq.append(obs)
        self.actionsDeq.append(action)
        self.rewardsDeq.append(reward)
        self.nextObssDeq.append(nextObs)
        self.terminalsDeq.append(terminal)
        if self.curr < self.maxSize:
            self.curr += 1
    
    def sample(self, elem):
        idxs = np.random.choice(np.amin([self.maxSize,self.curr]), elem , replace=False)

        obss = np.zeros((elem, self.obsDeq[0].shape[1]))
        actions = np.zeros((elem, self.actionsDeq[0].shape[1]))
        rewards = np.zeros((elem,))
        nextObss = np.zeros((elem, self.nextObssDeq[0].shape[1]))
        terminals = np.zeros((elem,))

        for i,e in enumerate(idxs):
            obss[i] = self.obsDeq[e]
            actions[i] = self.actionsDeq[e]
            rewards[i] = self.rewardsDeq[e]
            nextObss[i] = self.nextObssDeq[e]
            terminals[i] = self.terminalsDeq[e]

        return obss, actions, rewards, nextObss, terminals

