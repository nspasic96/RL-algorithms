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
    def __init__(self, env, normOb=True, centerRew=True, scaleRew=True, gamma=-1, clipOb=10., clipRew=10., episodicScale=True):
        super(EnvironmentWrapper, self).__init__(env)
        self.env = env
        self.normOb = normOb
        self.centerRew = centerRew
        self.scaleRew = scaleRew
        self.gamma = gamma
        if self.gamma > 0:
            self.ret = 0
        self.episodicScale = episodicScale
        self.clipOb = clipOb
        self.clipRew = clipRew
        if(normOb):
            self.obsRMS = RunningMeanStd(shape=self.observation_space.shape)
        if(centerRew or scaleRew or gamma > 0):
            self.rewRMS = RunningMeanStd(shape=(1,))
            
        assert self.gamma <= 0 or (self.centerRew == False and self.scaleRew == False)
    
    def step(self,action):
        if(self.gamma > 0):
            return self.clippingWithGamma(action)
        else:
            return self.normalClipping(action)
    """
    "Incorrect" reward normalization [copied from OAI code] Incorrect in the sense that we 
    1. update return
    2. divide reward by std(return) *without* subtracting and adding back mean
    """
    def clippingWithGamma(self, action):
        origObs, origRew, origTer, infos = self.env.step(action)

        obs = origObs
        rew = origRew
        ter = origTer

        if(self.normOb):
            self.obsRMS.update(origObs)
            obs = np.clip((origObs - self.obsRMS.mean)/(self.obsRMS.var + 1e-8),-self.clipOb,self.clipOb)
        
        self.ret = self.ret * self.gamma + rew
        self.rewRMS.update(np.array([self.ret]))  
        
        rew = rew / (self.rewRMS.var + 1e-8)
        
        if(self.clipRew > 0):
            rew = np.clip(rew,-self.clipRew,self.clipRew)[0]
            infos['origRew'] = origRew

        return obs, rew, ter, infos
    
    def normalClipping(self, action):
        origObs, origRew, origTer, infos = self.env.step(action)

        obs = origObs
        rew = origRew
        ter = origTer

        if(self.normOb):
            self.obsRMS.update(origObs)
            obs = np.clip((origObs - self.obsRMS.mean)/(self.obsRMS.var + 1e-8),-self.clipOb,self.clipOb)
        
        self.rewRMS.update(np.array([rew]))  
        if(self.centerRew):
            rew = rew - self.rewRMS.mean
        if(self.scaleRew):
            if(self.centerRew):
                rew = rew/(self.rewRMS.var + 1e-8)
            else:
                diff = rew - self.rewRMS.mean
                diff = diff/(self.rewRMS.std + 1e-8)
                rew = diff + self.rewRMS.mean            
        if(self.clipRew > 0):
            rew = np.clip(rew,-self.clipRew,self.clipRew)[0]
            infos['origRew'] = origRew

        return obs, rew, ter, infos

    def reset(self):
        obs = self.env.reset()
        
        if self.episodicScale:            
            if(self.normOb):
                self.obsRMS = RunningMeanStd(shape=self.observation_space.shape)
            if(self.centerRew or self.scaleRew or self.gamma > 0):
                self.rewRMS = RunningMeanStd(shape=(1,))
            self.rew = 0
            
        if(self.normOb):
            self.obsRMS.update(obs)
            obs = np.clip((obs - self.obsRMS.mean)/(self.obsRMS.var + 1e-8),-self.clipOb,self.clipOb)
        return obs

             
