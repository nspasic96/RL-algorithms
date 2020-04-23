import gym
import numpy as np
from utils import updateMeanVarCountFromMoments

class RunningMeanStd():
    def __init__(self, shape=()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4
    
    def update(self,x):
        batchMean = np.mean(x, axis=0)
        batchVar = np.var(x, axis=0)
        batchCount = x.shape[0]
        self.mean, self.var, self.count = updateMeanVarCountFromMoments(self.mean, self.var, self.count, batchMean, batchVar, batchCount)

class EnvironmentWrapper(gym.core.Wrapper):
    def __init__(self, env, normOb=True, normRew=True, clipOb=10., clipRew=10.):
        super(EnvironmentWrapper, self).__init__(env)
        self.env = env
        self.normOb = normOb
        self.normRew = normRew
        self.clipOb = clipOb
        self.clipRew = clipRew
        if(normOb):
            self.obsRMS = RunningMeanStd(shape=self.observation_space.shape)
        if(normRew):
            self.rewRMS = RunningMeanStd(shape=(1,))
    
    def step(self, action):
        origObs, origRew, origTer, infos = self.env.step(action)

        obs = origObs
        rew = origRew
        ter = origTer

        if(self.normOb):
            self.obsRMS.update(origObs)
            obs = np.clip((origObs - self.obsRMS.mean)/(self.obsRMS.var + 1e-8),-self.clipOb,self.clipOb)
            
        if(self.normRew):
            self.rewRMS.update(np.array([origRew]))
            rew = np.clip((origRew - self.rewRMS.mean)/(self.rewRMS.var + 1e-8),-self.clipRew,self.clipRew)[0]
            infos['origRew'] = origRew

        return obs, rew, ter, infos

    def reset(self):
        obs = self.env.reset()
        if(self.normOb):
            self.obsRMS.update(obs)
            obs = np.clip((obs - self.obsRMS.mean)/(self.obsRMS.var + 1e-8),-self.clipOb,self.clipOb)
        return obs

             
