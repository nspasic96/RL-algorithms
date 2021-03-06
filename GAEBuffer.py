import utils
import numpy as np


class GAEBuffer:
    
    def __init__(self, gamma, lamb, size, obsLen, actLen, advantageNorm, additionalInfoLengths):
        self.currentIdx, self.pathStartIdx = 0, 0
        self.gamma = gamma
        self.lamb = lamb

        self.obsBuff = np.zeros((size, obsLen))
        self.actBuff = np.zeros((size, actLen))
        self.predValsBuff = np.zeros((size,))
        self.samLogProbBuff = np.zeros((size,))
        self.rewardsBuff = np.zeros((size,))
        self.advantagesBuff = np.zeros((size,))
        self.returnsBuff = np.zeros((size,))
        self.advantageNorm=advantageNorm
        
        self.additionalInfos = [np.zeros((size, l)) if l > 1 else np.zeros((size,)) for l in additionalInfoLengths]
                   


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
        self.rewardsBuff[self.currentIdx] = reward
        self.predValsBuff[self.currentIdx] = predictedV
        self.samLogProbBuff[self.currentIdx] = logProbSampledAction
        for idx, additionalInfo in enumerate(additionalInfos):
            self.additionalInfos[idx][self.currentIdx] = additionalInfo 
            

        self.currentIdx += 1

    def finishPath(self, val=0):
        path_slice = slice(self.pathStartIdx, self.currentIdx)

        pathRews = np.append(self.rewardsBuff[path_slice], val)
        pathPredVals = np.append(self.predValsBuff[path_slice], val)

        deltas = pathRews[:-1] + self.gamma*pathPredVals[1:] - pathPredVals[:-1]        
        self.advantagesBuff[path_slice] = utils.disount_cumsum(deltas, self.gamma * self.lamb)
        self.returnsBuff[path_slice] = self.advantagesBuff[path_slice]+self.predValsBuff[path_slice]
        
        # Advantage normalization
        if self.advantageNorm:
            self.advantagesBuff[path_slice] = (self.advantagesBuff[path_slice] - self.advantagesBuff[path_slice].mean()) / (self.advantagesBuff[path_slice].std() + 1e-10)

        self.pathStartIdx = self.currentIdx 

    def get(self):

        self.currentIdx, self.pathStartIdx = 0, 0

        return self.obsBuff, self.actBuff, self.advantagesBuff, self.samLogProbBuff, self.returnsBuff, self.predValsBuff, self.additionalInfos
        
    