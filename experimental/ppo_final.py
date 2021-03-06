import argparse
import gym
import pybulletgym
import numpy as np
import tensorflow as tf
import time
import wandb
import os

import sys
sys.path.append("../")

import utils

from collections import deque
from EnvironmentWrapper import EnvironmentWrapper
from Statistics import Statistics
from gym.wrappers import TimeLimit
from stable_baselines.common.vec_env import DummyVecEnv

parser = argparse.ArgumentParser(description='PPO')

#general parameters
parser.add_argument('--gym-id', type=str, default="HopperPyBulletEnv-v0",
                   help='the id of the gym environment')
parser.add_argument('--seed', type=int,
                   help='seed of the experiment')
parser.add_argument('--epochs', type=int, default=500,
                   help="epochs to train")
parser.add_argument('--epoch_len', type=int, default=2048,
                   help="length of one epoch")
parser.add_argument('--max_episode_len', type=int, default=1000,
                   help="max length of one episode")
parser.add_argument('--gamma', type=float, default=0.99,
                   help='the discount factor gamma')
parser.add_argument('--lambd', type=float, default=0.95,
                   help='lambda for GAE-Lambda')
parser.add_argument('--learning_rate_state_value', type=float, default=3e-4,
                   help='learning rate of the optimizer of state value network')
parser.add_argument('--learning_rate_policy', type=float, default=4e-4,
                   help='learning rate of the optimizer of policy network')
parser.add_argument('--hidden_layers_state_value', type=int, nargs='+', default=[64,64],
                   help='hidden layers size in state value network')
parser.add_argument('--hidden_layers_policy', type=int, nargs='+', default=[64,64],
                   help='hidden layers size in policy network')

#WANDB parameters
parser.add_argument('--wandb_projet_name', type=str, default="proximal-policy-optimization",
                   help="the wandb's project name")
parser.add_argument('--wandb_log',  type=lambda x: (str(x).lower() == 'true'), default=False,
                   help='whether to log results to wandb')
parser.add_argument('--alg_name', type=str, default=None,
                   help="algorithm name, if remains none default values will be used")

#test parameters
parser.add_argument('--test_episodes_with_noise', type=int, default=100,
                   help='when testing, test_episodes_with_noise will be played and taken for calculting statistics')
parser.add_argument('--test_episodes_without_noise', type=int, default=100,
                   help='when testing, test_episodes_without_noise will be played and taken for calculting statistics')
parser.add_argument('--render', type=lambda x: (str(x).lower() == 'true'), default=False,
                   help='whether to render agent when it is being tested')

#PPO specific parameters
parser.add_argument('--update_epochs', type=int, default=10,
                   help="number of updates")
parser.add_argument('--minibatch_size', type=int, default=64,
                   help="batch size for policy network optimization")
parser.add_argument('--eps', type=float, default=0.2,
                   help='epsilon for clipping in objective')
parser.add_argument('--val_eps', type=float, default=-0.2,
                   help='epsilon for clipping value function, negative value to turn it off')#1
parser.add_argument('--reward_scaling', type=lambda x: (str(x).lower() == 'true'), default=False,
                   help='whether environment should transform reward before returning it')#2
parser.add_argument('--orthogonal_initialization', type=lambda x: (str(x).lower() == 'true'), default=False,
                   help='whether to initialize weights with orthogonal initializer')#3
parser.add_argument('--adam_eps', type=float, default=1e-8,
                   help="epsilon for adam")
parser.add_argument('--lr_annealing', type=lambda x: (str(x).lower() == 'true'), default=False,
                   help='whether to anneal Adam learning rate in every epoch')#4
parser.add_argument('--rew_clip', type=float, default=-10. ,
                   help='range in which to clip reward')#5
parser.add_argument('--obs_norm', type=lambda x: (str(x).lower() == 'true'), default=False,
                   help="whether to normalize observations")#6
parser.add_argument('--obs_clip', type=float, default=-10. ,
                   help='range in which to clip observations after normalizing')#7
parser.add_argument('--tanh_act', type=lambda x: (str(x).lower() == 'true'), default=False,
                   help='tanh activations if set to true, relu otherwise')#8
parser.add_argument('--grad_clip', type=float, default=-1. ,
                   help='clip gradient such that norm is equal to grad_clipping')#9
parser.add_argument('--norm_adv', type=lambda x: (str(x).lower() == 'true'), default=False,
                   help="whether to normalize batch of advantages obtained from GAE buffer for policy optimization")#10
args = parser.parse_args()

dtype = tf.float32
dtypeNp = np.float32

if not args.seed:
    args.seed = int(time.time())

graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    
    def makeEnvLambda(gym_id, seed, normOb, rewardNormalization, clipOb, clipRew, **kwargs):
        def func():
            env = gym.make(gym_id)
            env = TimeLimit(env, max_episode_steps = args.max_episode_len)
            env = EnvironmentWrapper(env.env, normOb=normOb, rewardNormalization=rewardNormalization, clipOb=clipOb, clipRew=clipRew, **kwargs)
            env.seed(args.seed)
            env.action_space.seed(args.seed)
            env.observation_space.seed(args.seed)  
            return env
        return func
    
    rewardNormalization = "returns" if args.reward_scaling else None
    clipOb = 100000. if args.obs_clip < 0 else args.obs_clip
    clipRew = 100000. if args.rew_clip < 0 else args.rew_clip    
    
    env = DummyVecEnv([makeEnvLambda(args.gym_id, args.seed, normOb=args.obs_norm, rewardNormalization=rewardNormalization, clipOb=clipOb, clipRew=clipRew, gamma=args.gamma)])
      
    np.random.seed(args.seed)  
    tf.set_random_seed(args.seed)

    discreteActionsSpace = utils.is_discrete(env)
    
    inputLength = env.observation_space.shape[0]
    outputLength = env.action_space.n if discreteActionsSpace else env.action_space.shape[0]
    
    #summeries placeholders and summery scalar objects
    epRewTestPh = tf.placeholder(tf.float32, shape=None, name='episode_test_real_reward_mean_summary')
    epRewTrainPh = tf.placeholder(tf.float32, shape=None, name='episode_train_real_reward_latest_mean_summary')
    epTotalRewPh = tf.placeholder(tf.float32, shape=None, name='episode_reward_train_summary')
    epLenPh = tf.placeholder(tf.float32, shape=None, name='episode_length_train_summary')
    SVLossPh = tf.placeholder(tf.float32, shape=None, name='value_function_loss_summary')
    LlossNewPh = tf.placeholder(tf.float32, shape=None, name='surrogate_function_value_summary')
    KLPh = tf.placeholder(dtype, shape=None, name='kl_divergence_summary')
    KLFirstPh = tf.placeholder(dtype, shape=None, name='kl_divergence_first_summary')
    epRewLatestMeanTestSum = tf.summary.scalar('episode_test_reward_mean', epRewTestPh)
    epRewLatestMeanTrainSum = tf.summary.scalar('episode_train_reward_mean', epRewTrainPh)
    epTotalRewSum = tf.summary.scalar('episode_reward_train', epTotalRewPh)
    epLenSum = tf.summary.scalar('episode_length_train', epLenPh)
    SVLossSummary = tf.summary.scalar('value_function_loss', SVLossPh)
    LlossNewSum = tf.summary.scalar('surrogate_function_value', LlossNewPh)
    KLSum = tf.summary.scalar('kl_divergence', KLPh)      
    KLFirstSum = tf.summary.scalar('kl_divergence_first', KLFirstPh)      
    
    implSuffix = os.path.basename(__file__).rstrip(".py")
    experimentName = f"{args.gym_id}__{implSuffix}__{args.seed}__{int(time.time())}"
    writer = tf.summary.FileWriter(f"runs/{experimentName}", graph = sess.graph)
    
    if args.wandb_log:
        
        cnf = vars(args)
        cnf['action_space_type'] = 'discrete' if discreteActionsSpace else 'continuous'
        cnf['input_length'] = inputLength
        cnf['output_length'] = outputLength
        cnf['exp_name_tb'] = experimentName
        
        if args.alg_name is None:
            cnf['alg_name'] = "PPO-No-Name"
        else:
            cnf['alg_name'] = args.alg_name
            
        wandb.init(project=args.wandb_projet_name, config=cnf, name=experimentName, tensorboard=True)   
    
    #definition of placeholders
    logProbSampPh = tf.placeholder(dtype = tf.float32, shape=[None], name="logProbSampled") #log probabiliy of action sampled from sampling distribution (pi_old)
    advPh = tf.placeholder(dtype = tf.float32, shape=[None], name="advantages") #advantages obtained using GAE-lambda and values obtainet from StateValueNetwork V
    VPrevPh = tf.placeholder(dtype = dtype, shape=[None], name="previousValues") #values for previous iteration returned by StateValueNetwork V
    totalEstimatedDiscountedRewardPh = tf.placeholder(dtype = dtype, shape=[None], name="totalEstimatedDiscountedReward") #total discounted cumulative reward estimated as advantage + previous values
    trainableParamsFlatten = tf.placeholder(dtype = dtype, shape=[None], name = "trainableParams") #policy params flatten, used in assingment of pi params if KL rollback is enabled
    obsPh = tf.placeholder(dtype=tf.float32, shape=[None, inputLength], name="observations") #observations
    learningRateVfPh = tf.placeholder(dtype=dtype, shape=[], name="learningRateVfPh")
    learningRatePolicyPh = tf.placeholder(dtype=dtype, shape=[], name="learningRatePolicyPh")
        
    if discreteActionsSpace:
        aPh = tf.placeholder(dtype=tf.int32, shape=[None], name="actions") #actions taken
        logProbsAllPh = tf.placeholder(dtype= tf.float32, shape=[None, outputLength], name="logProbsAll") #log probabilities of all actions according to sampling distribution (pi_old)
        additionalInfoLengths = [outputLength]
    else:
        aPh = tf.placeholder(dtype=tf.float32, shape=[None,outputLength], name="actions")
        oldActionMeanPh = tf.placeholder(dtype=tf.float32, shape=[None,outputLength], name="actionsMeanOld")
        oldActionLogStdPh = tf.placeholder(dtype=tf.float32, shape=[None,outputLength], name="actionsLogStdOld") 
        additionalInfoLengths = [outputLength, outputLength]
    
    #definition of networks
    with tf.variable_scope("AllTrainableParams"):
        if discreteActionsSpace:
            #TODO            
            KLcontraint = utils.categorical_kl(logProbWithCurrParamsOp, logProbsAllPh) 
        else:
            activation = tf.nn.tanh if args.tanh_act else tf.nn.relu
            initializationHidden = tf.orthogonal_initializer(2**0.5) if args.orthogonal_initialization else tf.contrib.layers.xavier_initializer()
            initializationFinalValue = tf.orthogonal_initializer(1) if args.orthogonal_initialization else tf.contrib.layers.xavier_initializer()
            initializationFinalPolicy = tf.orthogonal_initializer(0.01) if args.orthogonal_initialization else tf.contrib.layers.xavier_initializer()
            
            #value network
            curNode = tf.layers.Dense(args.hidden_layers_state_value[0], activation, kernel_initializer=initializationHidden, name="fc1")(obsPh)
            for i,l in enumerate(args.hidden_layers_state_value[1:]):
                curNode = tf.layers.Dense(l, activation, kernel_initializer=initializationHidden, name="fc{}".format(i+2))(curNode)
                
            vfOutputOp = tf.squeeze(tf.layers.Dense(1, kernel_initializer=initializationFinalValue, name="outputV")(curNode),1)
            
            #policy network
            curNode = tf.layers.Dense(args.hidden_layers_policy[0], activation, kernel_initializer=initializationHidden, name="fc1")(obsPh)
            for i,l in enumerate(args.hidden_layers_policy[1:]):
                curNode = tf.layers.Dense(l, activation, kernel_initializer=initializationHidden, name="fc{}".format(i+2))(curNode)
               
            actionMeanOp = tf.layers.Dense(outputLength, kernel_initializer=initializationFinalPolicy, name="outputA")(curNode)
            actionLogStdOp = tf.get_variable(name="ActionsLogStdDetachedTrainable", initializer=-0.3*np.ones((1, outputLength), dtype=np.float32), trainable=True)
            actionStdOp = tf.math.exp(actionLogStdOp)                
            actionFinalOp = actionMeanOp + tf.random_normal(tf.shape(actionMeanOp)) * actionStdOp 
            sampledLogProbsOp = utils.gaussian_likelihood(actionFinalOp, actionMeanOp, actionLogStdOp)
            logProbWithCurrParamsOp = utils.gaussian_likelihood(aPh, actionMeanOp, actionLogStdOp)  
            
            KLcontraint = utils.diagonal_gaussian_kl(actionMeanOp, actionLogStdOp, oldActionMeanPh, oldActionLogStdPh)   
    
    #definition of losses to optimize
    ratio = tf.exp(logProbWithCurrParamsOp - logProbSampPh)
    clippedRatio = tf.clip_by_value(ratio, 1-args.eps, 1+args.eps)
    Lloss = -tf.reduce_mean(tf.minimum(ratio*advPh,clippedRatio*advPh)) # - sign because we want to maximize our objective
    
    if args.val_eps > 0:
        vLossUncliped = (vfOutputOp - totalEstimatedDiscountedRewardPh)**2
        vClipped = VPrevPh + tf.clip_by_value(vfOutputOp - VPrevPh, -args.val_eps, args.val_eps)
        vLossClipped = (vClipped - totalEstimatedDiscountedRewardPh)**2
        vLossMax = tf.maximum(vLossClipped, vLossUncliped)
        stateValueLoss = tf.reduce_mean(0.5 * vLossMax)
    else:
        stateValueLoss = tf.reduce_mean((vfOutputOp - totalEstimatedDiscountedRewardPh)**2)
               
    #create optimizers depending on lr annealing option
    if args.lr_annealing:                
        optimizatierVf = tf.train.AdamOptimizer(learning_rate = learningRateVfPh, epsilon=args.adam_eps)
        optimizatierPolicy = tf.train.AdamOptimizer(learning_rate = learningRatePolicyPh, epsilon=args.adam_eps)
    else:
        optimizatierVf = tf.train.AdamOptimizer(learning_rate = args.learning_rate_state_value, epsilon=args.adam_eps)        
        optimizatierPolicy = tf.train.AdamOptimizer(learning_rate = args.learning_rate_policy, epsilon=args.adam_eps)
    
    #create optimization step dependingo on grad_clip value
    if args.grad_clip > 0:
        valGradients, valVaribales = zip(*optimizatierVf.compute_gradients(stateValueLoss))  
        valGradients, _ = tf.clip_by_global_norm(valGradients, args.grad_clip)       
        optimizationStepVf = optimizatierVf.apply_gradients(zip(valGradients, valVaribales))
        
        polGradients, polVaribales = zip(*optimizatierPolicy.compute_gradients(Lloss))  
        polGradients, _ = tf.clip_by_global_norm(polGradients, args.grad_clip)       
        optimizationStepPolicy = optimizatierPolicy.apply_gradients(zip(polGradients, polVaribales))
    else:
         optimizationStepVf = optimizatierVf.minimize(stateValueLoss)
         optimizationStepPolicy = optimizatierPolicy.minimize(Lloss)
       
    trainableParams = utils.get_vars("AllTrainableParams")
    getTrainableParams = utils.flat_concat(trainableParams)
    setTrainableParams = utils.assign_params_from_flat(trainableParamsFlatten, trainableParams)
    
    #tf session initialization
    init = tf.initialize_local_variables()
    init2 = tf.initialize_all_variables()
    sess.run([init,init2])
    
    nextObs = env.reset() 
    nextDone = 0      
    epLen = 0
    epTotalRew = 0
    epTotalTrainRews = deque(maxlen = args.test_episodes_with_noise)  
    

    #algorithm
    for e in range(args.epochs):
        print("Epoch {} started".format(e))
        
        obs = np.zeros((args.epoch_len,inputLength))
        rewards = np.zeros((args.epoch_len,))
        dones = np.zeros((args.epoch_len,))
        predVals = np.zeros((args.epoch_len,))
        actions = np.zeros((args.epoch_len, outputLength))
        sampledLogProb = np.zeros((args.epoch_len,))
        additionalInfos = []
        additionalInfos.append(np.zeros((args.epoch_len,outputLength)))#for action means
        additionalInfos.append(np.zeros((args.epoch_len,outputLength)))#for log stds
        
        epLens = []  
        
        epochSt = time.time()
        for l in range(args.epoch_len):
            
            obs[l] = nextObs.copy() 
            dones[l] = nextDone
            
            if discreteActionsSpace:
                #this needs fixing
                sampledAction, logProbSampledAction, logProbsAll = policy.getSampledActions(obs[l])
                additionalInfos = [logProbsAll]
            else:
                sampledAction, logProbSampledAction, actionsMean, actionLogStd = sess.run([actionFinalOp, sampledLogProbsOp, actionMeanOp, actionLogStdOp], feed_dict = {obsPh : np.expand_dims(obs[l],0)})
                additionalInfos[0][l] = actionsMean
                additionalInfos[1][l] = actionLogStd           
            nextObss, rews, nextDones, infoss = env.step(sampledAction) 
            nextObs, rewards[l], nextDone, infos = nextObss[0], rews[0], nextDones[0], infoss[0]
            sampledLogProb[l] = logProbSampledAction[0]
            
            if dones[l]:
                
                summaryRet, summaryLen = sess.run([epTotalRewSum, epLenSum], feed_dict = {epTotalRewPh:epTotalRew, epLenPh:epLen})
                globalStep = e*args.epoch_len + l
                writer.add_summary(summaryRet, globalStep)
                writer.add_summary(summaryLen, globalStep)
                
                epTotalTrainRews.append(epTotalRew)
                epLen = 0
                epTotalRew = 0
                
            epLen += 1
            epTotalRew += infos["origRew"]   
            actions[l] = sampledAction[0]
              
        simulationEnd = time.time()      
        print("\tSimulation in epoch {} finished in {}".format(e, simulationEnd-epochSt))
       
        predVals = sess.run(vfOutputOp, feed_dict = {obsPh : obs})
                
        #calculating advantages
        lastValue = sess.run(vfOutputOp, feed_dict={obsPh:np.expand_dims(nextObs,0)})
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(args.epoch_len)):
            if t == args.epoch_len - 1:
                nextNonTerminal = 1.0 - nextDone
                nextValue = lastValue
            else:
                nextNonTerminal = 1.0 - dones[t+1]
                nextValue = predVals[t+1]
            delta = rewards[t] + args.gamma * nextValue * nextNonTerminal - predVals[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.lambd * nextNonTerminal * lastgaelam
            
        returns = advantages + predVals
 
        learningRatePolicy = utils.annealedNoise(args.learning_rate_policy, 0, args.epochs, e)
        learningRateVf = utils.annealedNoise(args.learning_rate_state_value, 0, args.epochs, e)
    
        #update
        updateStart = time.time()      
        total = args.epoch_len
        for j in range(args.update_epochs):
            perm = np.random.permutation(total)
            start = 0
            #oldParams = sess.run(getTrainableParams)
            
            while(start < total):    
                end = np.amin([start+args.minibatch_size, total])
                
                sess.run(optimizationStepPolicy, feed_dict={obsPh : obs[perm[start:end]], aPh: actions[perm[start:end]], VPrevPh:predVals[perm[start:end]], advPh : utils.normalize(advantages[perm[start:end]]) if args.norm_adv else advantages[perm[start:end]], logProbSampPh : sampledLogProb[perm[start:end]], learningRatePolicyPh:learningRatePolicy})
                sess.run(optimizationStepVf, feed_dict={obsPh : obs[perm[start:end]], totalEstimatedDiscountedRewardPh : returns[perm[start:end]], VPrevPh:predVals[perm[start:end]], learningRateVfPh:learningRateVf})

                start = end
                
            if j==0:#log just first update when ratios are 1
                klFirst = sess.run(KLcontraint, feed_dict={obsPh : obs, oldActionMeanPh : additionalInfos[0], oldActionLogStdPh : additionalInfos[1]})
                summaryKLFirst = sess.run(KLFirstSum, feed_dict={KLFirstPh:klFirst})
                writer.add_summary(summaryKLFirst ,e)   
            if j == args.update_epochs - 1:
                kl = sess.run(KLcontraint, feed_dict={obsPh : obs, oldActionMeanPh : additionalInfos[0], oldActionLogStdPh : additionalInfos[1]})
                summaryKL = sess.run(KLSum, feed_dict={KLPh:kl})
                writer.add_summary(summaryKL, e)
    
    
        updateEnd = time.time()      
        
        print("\tUpdate in epoch {} updated in {}".format(e, updateEnd-updateStart))    
         
        LlossOld = sess.run(Lloss , feed_dict={obsPh : obs, aPh: actions, advPh : advantages, logProbSampPh : sampledLogProb})#, logProbsAllPh : allLogProbs})#L function inputs: observations, advantages estimated, logProb of sampled action, logProbsOfAllActions
        SVLoss = sess.run(stateValueLoss, feed_dict={obsPh : obs, totalEstimatedDiscountedRewardPh : returns, VPrevPh:predVals})  
                    
        if args.wandb_log:
            summarySVm, summaryLloss = sess.run([SVLossSummary, LlossNewSum], feed_dict = {SVLossPh:SVLoss, LlossNewPh:LlossOld})
            writer.add_summary(summarySVm, (e+1)*args.epoch_len)
            writer.add_summary(summaryLloss, (e+1)*args.epoch_len)
        
        epochEnd = time.time()
        print("Epoch {} ended in {}".format(e, epochEnd-epochSt))
        
    summaryLatestTrainRet = sess.run(epRewLatestMeanTrainSum, feed_dict = {epRewTrainPh:np.mean(epTotalTrainRews)})
    writer.add_summary(summaryLatestTrainRet)   
    if args.wandb_log:
        wandb.log({"episode_train_reward_mean" : np.mean(epTotalTrainRews)})
        
    print("Testing agent without noise for {} episodes after training".format(args.test_episodes_without_noise))
    osbTest = env.reset()
    testRets = []
    testRet = 0        
    while len(testRets) < args.test_episodes_without_noise+1:
        
        if args.render:
            env.envs[0].render() 
            
        sampledActionsTest = sess.run(actionMeanOp, feed_dict = {obsPh : osbTest}) 
        nextObss, _, nextDones, infoss = env.step(sampledActionsTest) 
          
        testRet += infoss[0]["origRew"]
        osbTest = nextObss.copy()  
        
        if nextDones[0]:
            testRets.append(testRet)
            testRet = 0
        
        
    meanLatest = np.mean(testRets[1:])
    summaryLatestRet = sess.run(epRewLatestMeanTestSum, feed_dict = {epRewTestPh:meanLatest})
    writer.add_summary(summaryLatestRet)    
    
    writer.close()
    env.close()
    