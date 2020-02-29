import utils
import numpy as np


class GAEBuffer:
    
    def __init__(self, gamma, lamb, size, obsLen, actLen):
        self.currentIdx, self.pathStartIdx = 0, 0
        self.gamma = gamma
        self.lamb = lamb

        self.obsBuff = np.zeros((size, obsLen))
        self.actBuff = np.zeros(size, dtype=np.int32)
        self.predValsBuff = np.zeros(size)
        self.samLogProbBuff = np.zeros(size)
        self.allLogProbBuff = np.zeros((size, actLen))
        self.rewardsBuff = np.zeros(size)
        self.advantagesBuff = np.zeros(size)
        self.returnsBuff = np.zeros(size)



    def add(self, obs, action, predictedV, logProbSampledAction, logProbsAll, reward):
        self.obsBuff[self.currentIdx] = obs
        self.actBuff[self.currentIdx] = action
        self.predValsBuff[self.currentIdx] = predictedV
        self.samLogProbBuff[self.currentIdx] = logProbSampledAction
        self.allLogProbBuff[self.currentIdx] = logProbsAll
        self.rewardsBuff[self.currentIdx] = reward

        self.currentIdx += 1

    def finishPath(self, val=0):
        path_slice = slice(self.pathStartIdx, self.currentIdx)

        pathRews = np.append(self.rewardsBuff[path_slice], val)
        pathPredVals = np.append(self.predValsBuff[path_slice], val)

        deltas = pathRews[:-1] + self.gamma*pathPredVals[1:] - pathPredVals[:-1]        
        self.advantagesBuff[path_slice] = utils.disount_cumsum(deltas, self.gamma * self.lamb) 
        self.returnsBuff[path_slice] = utils.disount_cumsum(pathRews, self.gamma)[:-1]

        self.currentIdx = self.pathStartIdx

    def get(self):

        self.currentIdx, self.pathStartIdx = 0, 0

        observations = self.obsBuff
        actions = self.actBuff
        sampledLogProb = self.samLogProbBuff
        allLogProbs = self.allLogProbBuff
        advantages = self.advantagesBuff
        returns = self.returnsBuff


        return observations, actions, advantages, sampledLogProb, allLogProbs, returns
        
    