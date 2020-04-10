import utils
import numpy as np


class GAEBuffer:
    
    def __init__(self, gamma, lamb, size, obsLen, actLen, additionalInfoLengths):
        self.currentIdx, self.pathStartIdx = 0, 0
        self.gamma = gamma
        self.lamb = lamb

        self.obsBuff = np.zeros((size, obsLen))
        self.actBuff = np.zeros((size, actLen))
        self.predValsBuff = np.zeros(size)
        self.samLogProbBuff = np.zeros(size)
        self.rewardsBuff = np.zeros(size)
        self.advantagesBuff = np.zeros(size)
        self.returnsBuff = np.zeros(size)
        
        self.additionalInfos = [np.zeros((size, l)) for l in additionalInfoLengths]


    def add(self, obs, action, predictedV, logProbSampledAction, reward, additionalInfos):
        """
        print(obs.shape)
        print(action.shape)
        print(predictedV.shape)
        print(logProbSampledAction.shape)
        print(logProbsAll.shape)
        print(reward.shape)
        
        if(self.currentIdx == 1):
            print(self.obsBuff)
            print(self.actBuff)
            print(self.rewardsBuff)
            print(self.predValsBuff)
            print(self.samLogProbBuff)
        """
            
        self.obsBuff[self.currentIdx] = obs
        self.actBuff[self.currentIdx] = action
        self.predValsBuff[self.currentIdx] = predictedV
        self.samLogProbBuff[self.currentIdx] = logProbSampledAction
        self.rewardsBuff[self.currentIdx] = reward
        for idx, additionalInfo in enumerate(additionalInfos):
            self.additionalInfos[idx][self.currentIdx] = additionalInfo

        self.currentIdx += 1

    def finishPath(self, val=0):
        path_slice = slice(self.pathStartIdx, self.currentIdx)

        pathRews = np.append(self.rewardsBuff[path_slice], val)
        pathPredVals = np.append(self.predValsBuff[path_slice], val)

        deltas = pathRews[:-1] + self.gamma*pathPredVals[1:] - pathPredVals[:-1]        
        self.advantagesBuff[path_slice] = utils.disount_cumsum(deltas, self.gamma * self.lamb) 
        self.returnsBuff[path_slice] = utils.disount_cumsum(pathRews, self.gamma)[:-1]

        self.pathStartIdx = self.currentIdx 

    def get(self):

        self.currentIdx, self.pathStartIdx = 0, 0

        return self.obsBuff, self.actBuff, self.advantagesBuff, self.samLogProbBuff, self.returnsBuff, self.additionalInfos
        
    