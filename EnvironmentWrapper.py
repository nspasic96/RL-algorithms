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
    """
    There are 3 values that reward normalization can take:
        1. none : Do Nothing, return the reward as it is
        2. returns : Track auxiliary current reward(self.ret) and update it in every step by multiplying old value with gamma and adding 
            new "raw" reward to it. Reward returen by environment wrapper is diveded by variance of all auxiliary current rewards
        3. rewards : subtract mean and divide by variance(both are optional and can be set with parameters, defaults are True) 
    """
    def __init__(self, env, normOb=True, rewardNormalization=None, clipOb=10., clipRew=10., episodicMeanVarObs=False, episodicMeanVarRew=False, **kwargs):
        
        super(EnvironmentWrapper, self).__init__(env)
        self.env = env
        self.normOb = normOb
        self.rewardNormalization = rewardNormalization
        self.clipOb = clipOb
        self.clipRew = clipRew
        self.episodicMeanVarObs = episodicMeanVarObs
        self.episodicMeanVarRew = episodicMeanVarRew
        
        self.obsRMS = RunningMeanStd(shape=self.observation_space.shape)
        self.rewRMS = RunningMeanStd(shape=(1,))
         
        assert (self.rewardNormalization is None or self.rewardNormalization in ["returns", "rewards"])
        if self.rewardNormalization is not None:
            if self.rewardNormalization == "returns": 
                self.gamma = kwargs.get('gamma',0.99)
                self.ret = 0       
            elif self.rewardNormalization == "rewards": 
                self.centerRew = kwargs.get('centerRew',True)  
                self.scaleRew = kwargs.get('scaleRew',True)      
    
    def step(self,action):
        
        origObs, origRew, origTer, infos = self.env.step(action)

        obs = origObs
        rew = origRew
        infos["origRew"] = rew
                
        if(self.normOb):
            self.obsRMS.update(origObs)
            obs = np.clip((origObs - self.obsRMS.mean)/(self.obsRMS.var + 1e-8),-self.clipOb,self.clipOb)

        if(self.rewardNormalization == "returns"):
            rew = self.rewardReturns(rew, infos)
        elif(self.rewardNormalization == "rewards"):
            rew = self.rewardRewards(rew, infos)

        if origTer:
            self.ret = 0
        
        return obs, rew, origTer, infos
    
    """
    "Incorrect" reward normalization [copied from OAI code] Incorrect in the sense that we 
    1. update return
    2. divide reward by std(return) *without* subtracting and adding back mean
    """
    def rewardReturns(self, rew, infos):
        
        self.ret = self.ret * self.gamma + rew
        self.rewRMS.update(np.array([self.ret]).copy())  
        
        rew = rew / (self.rewRMS.var + 1e-8)
        
        if(self.clipRew > 0):
            rew = np.clip(rew,-self.clipRew,self.clipRew)[0]

        return rew
    
    def rewardRewards(self, rew, infos):
        
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

        return rew

    def reset(self):
        obs = self.env.reset()
        
        if self.episodicMeanVarObs:            
            self.obsRMS = RunningMeanStd(shape=self.observation_space.shape)
        if(self.episodicMeanVarRew):
            self.rewRMS = RunningMeanStd(shape=(1,))
            
        if(self.normOb):
            self.obsRMS.update(obs)
            obs = np.clip((obs - self.obsRMS.mean)/(self.obsRMS.var + 1e-8),-self.clipOb,self.clipOb)

        return obs

             
