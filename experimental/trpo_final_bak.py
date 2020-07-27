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

parser = argparse.ArgumentParser(description='TRPO')

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
parser.add_argument('--learning_rate_state_value', type=float, default=1e-3,
                   help='learning rate of the optimizer of state-value function')
parser.add_argument('--learning_rate_policy', type=float, default=3e-4,
                   help='learning rate of the optimizer of policy function')
parser.add_argument('--hidden_layers_state_value', type=int, nargs='+', default=[64,64],
                   help='hidden layers size in state value network')
parser.add_argument('--hidden_layers_policy', type=int, nargs='+', default=[64,64],
                   help='hidden layers size in policy network')
parser.add_argument('--policy_network_batch_size', type=int, default=64,
                   help="batch size for policy network optimization")
parser.add_argument('--value_function_batch_size', type=int, default=64,
                   help='batch size for value function network')

#WANDB parameters
parser.add_argument('--wandb_projet_name', type=str, default="trust-region-policy-optimization",
                   help="the wandb's project name")
parser.add_argument('--wandb_log', type=lambda x: (str(x).lower() == 'true'), default=False,
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

#TRPO specific parameters
parser.add_argument('--delta', type=float, default=0.01,
                   help='max KL distance between two successive distributions at the start of anneailng')
parser.add_argument('--delta_final', type=float, default=0.01,
                   help='max KL distance between two successive distributions at the end of annealing')
parser.add_argument('--state_value_network_updates', type=int, default=10,
                   help="number of updates for state-value network")
parser.add_argument('--fisher_fraction', type=float, default=1,
                   help="fraction of data that is used to estimate Fisher Information Matrix")
parser.add_argument('--damping_coef', type=float, default=0.1,
                   help='damping coef')
parser.add_argument('--cg_iters', type=int, default=10,
                   help='number of iterations in cg algorithm')
parser.add_argument('--max_iters_line_search', type=int, default=10,
                   help="maximum steps to take in line serach")
parser.add_argument('--alpha', type=float, default=0.8,
                   help="defult step size in line serach")
parser.add_argument('--adam_eps', type=float, default=1e-5,
                   help="epsilon for adam")
parser.add_argument('--orig_returns', type=lambda x: (str(x).lower() == 'true'), default=False,
                   help="whether to calculate returns using only rewards from environment(when set to True) or using previous predictions and advantages")
parser.add_argument('--norm_adv', type=lambda x: (str(x).lower() == 'true'), default=False,
                   help="whether to normalize batch of advantages obtained from GAE buffer for policy optimization")
parser.add_argument('--norm_obs', type=lambda x: (str(x).lower() == 'true'), default=True,
                   help="whether to normalize observations")
parser.add_argument('--tanh_act', type=lambda x: (str(x).lower() == 'true'), default=True,
                   help="whether to use tanh as activation function")

#parameters related to TRPO+
parser.add_argument('--plus', type=lambda x: (str(x).lower() == 'true'), default=False,
                   help='whether to add code optimizations 1-4')
parser.add_argument('--plus_eps', type=float, default=-0.2,
                   help='epsilon for clipping value function, negative value to turn it off')
parser.add_argument('--plus_returns', type=lambda x: (str(x).lower() == 'true'), default=True,
                   help='whether to return returns instead of reward for training')
parser.add_argument('--plus_initialization', type=lambda x: (str(x).lower() == 'true'), default=True,
                   help='whether to initialize weights with orthogonal initializer')
parser.add_argument('--plus_lr_annealing', type=lambda x: (str(x).lower() == 'true'), default=False,
                   help='whether to anneal Adam learning rate in every epoch')

parser.add_argument('--plus_plus', type=lambda x: (str(x).lower() == 'true'), default=False,
                   help='whether to add code optimizations 5-9')
parser.add_argument('--plus_plus_reward_clipping', type=float, default=10. ,
                   help='range in which to clip reward')
parser.add_argument('--plus_plus_observation_normalization', type=lambda x: (str(x).lower() == 'true'), default=True,
                   help='whether to normalize observations with running mean and std')
parser.add_argument('--plus_plus_observation_clipping', type=float, default=10. ,
                   help='range in which to clip observations after normalizing(if set to True)')
parser.add_argument('--plus_plus_tanh', type=lambda x: (str(x).lower() == 'true'), default=True,
                   help='tanh activations if set to true, relu otherwise')
parser.add_argument('--plus_plus_grad_clip', type=float, default=1. ,
                   help='gradient l2 norm, negative value to turn it off')


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
           
    clipOb=1000000.
    clipRew=1000000
    rewardNormalization=None
    normOb=args.norm_obs
    if args.plus:
        if args.plus_returns:
            rewardNormalization = "returns"        
    if args.plus_plus:
        normOb = args.plus_plus_observation_normalization
        if args.plus_plus_observation_clipping > 0:
            clipOb = args.plus_plus_observation_clipping
        if args.plus_plus_reward_clipping > 0:
            clipRew = args.plus_plus_reward_clipping

    
    env = DummyVecEnv([makeEnvLambda(args.gym_id, args.seed, normOb=normOb, rewardNormalization=rewardNormalization, clipOb=clipOb, clipRew=clipRew, gamma=args.gamma)])    
           
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    discreteActionsSpace = utils.is_discrete(env)
    
    inputLength = env.observation_space.shape[0]
    outputLength = env.action_space.n if discreteActionsSpace else env.action_space.shape[0]
    
    #summeries placeholders and summery scalar objects      
    epRewTestPh = tf.placeholder(tf.float32, shape=None, name='episode_test_real_reward_latest_mean_summary')
    epRewTrainPh = tf.placeholder(tf.float32, shape=None, name='episode_train_real_reward_latest_mean_summary')
    epTotalRewPh = tf.placeholder(dtype, shape=None, name='episode_reward_train_summary')
    epLenPh = tf.placeholder(dtype, shape=None, name='episode_length_train_summary')
    SVLossPh = tf.placeholder(dtype, shape=None, name='value_function_loss_summary')
    SurrogateDiffPh = tf.placeholder(dtype, shape=None, name='surrogate_function_value_summary')
    KLPh = tf.placeholder(dtype, shape=None, name='kl_divergence_summary')
    epRewLatestMeanTestSum = tf.summary.scalar('episode_test_reward_mean', epRewTestPh)
    epRewLatestMeanTrainSum = tf.summary.scalar('episode_train_reward_mean', epRewTrainPh)
    epTotalRewSum = tf.summary.scalar('episode_reward_train', epTotalRewPh)
    epLenSum = tf.summary.scalar('episode_length_train', epLenPh)
    SVLossSummary = tf.summary.scalar('value_function_loss', SVLossPh)
    SurrogateDiffSum = tf.summary.scalar('surrogate_function_value', SurrogateDiffPh)
    KLSum = tf.summary.scalar('kl_divergence', KLPh)  
    
    #logging details      
    implSuffix = os.path.basename(__file__).rstrip(".py")
    prefix = ""
    if args.plus:
        prefix = prefix + "plus-"
    if args.plus_plus:
        prefix = prefix + "plus-plus-"
    experimentName = f"{prefix}{args.gym_id}__{implSuffix}__{args.seed}__{int(time.time())}"
    writer = tf.summary.FileWriter(f"runs/{experimentName}", graph = sess.graph)
    
    if args.wandb_log:
        
        cnf = vars(args)
        cnf['action_space_type'] = 'discrete' if discreteActionsSpace else 'continuous'
        cnf['input_length'] = inputLength
        cnf['output_length'] = outputLength
        cnf['exp_name_tb'] = experimentName
        
        if args.alg_name is None:
            cnf['alg_name'] = "TRPO+" if (args.plus or args.plus_plus) else "TRPO"
        else:
            cnf['alg_name'] = args.alg_name
        wandb.init(project=args.wandb_projet_name, config=cnf, name=experimentName, tensorboard=True)
        
    #definition of placeholders
    logProbSampPh = tf.placeholder(dtype = dtype, shape=[None], name="logProbSampled") #log probabiliy of action sampled from sampling distribution (pi_old)
    advPh = tf.placeholder(dtype = dtype, shape=[None], name="advantages") #advantages obtained using GAE-lambda and values obtainet from StateValueNetwork V
    VPrevPh = tf.placeholder(dtype = dtype, shape=[None], name="previousValues") #values for previous iteration returned by StateValueNetwork V
    totalEstimatedDiscountedRewardPh = tf.placeholder(dtype = dtype, shape=[None], name="totalDiscountedReward") #total discounted cumulative reward
    policyParamsFlatten= tf.placeholder(dtype = dtype, shape=[None], name = "policyParams") #policy params flatten, used in assingment of pi params in line search algorithm
    obsPh = tf.placeholder(dtype=dtype, shape=[None, inputLength], name="observations") #observations 
    learningRatePh = tf.placeholder(dtype=dtype, shape=[], name="learningRatePh")#learning rate placeholder, used when TRPO+ is enabled
    
    if discreteActionsSpace:
        aPh = tf.placeholder(dtype=tf.int32, shape=[None,1], name="actions") #actions taken
        logProbsAllPh = tf.placeholder(dtype= dtype, shape=[None, outputLength], name="logProbsAll") #log probabilities of all actions according to sampling distribution (pi_old)
        additionalInfoLengths = [outputLength]
    else:
        aPh = tf.placeholder(dtype=dtype, shape=[None,outputLength], name="actions")
        oldActionMeanPh = tf.placeholder(dtype=dtype, shape=[None,outputLength], name="actionsMeanOld")
        oldActionLogStdPh = tf.placeholder(dtype=dtype, shape=[None,outputLength], name="actionsLogStdOld") 
        additionalInfoLengths = [outputLength, outputLength]
    
    #definition of networks        
    activation = tf.nn.tanh if args.tanh_act else tf.nn.relu
    initializationHidden = tf.orthogonal_initializer(2**0.5) if (args.plus and args.plus_initialization) else tf.contrib.layers.xavier_initializer()
    initializationFinalValue = tf.orthogonal_initializer(1) if (args.plus and args.plus_initialization) else tf.contrib.layers.xavier_initializer()
    initializationFinalPolicy = tf.orthogonal_initializer(0.01) if (args.plus and args.plus_initialization) else tf.contrib.layers.xavier_initializer()
    
    #value network    
    with tf.variable_scope("StateValueNetwork"):
        curNode = tf.layers.Dense(args.hidden_layers_state_value[0], activation, kernel_initializer=initializationHidden, name="fc1")(obsPh)
        for i,l in enumerate(args.hidden_layers_state_value[1:]):
            curNode = tf.layers.Dense(l, activation, kernel_initializer=initializationHidden, name="fc{}".format(i+2))(curNode)
            
        vfOutputOp = tf.squeeze(tf.layers.Dense(1, kernel_initializer=initializationFinalValue, name="outputV")(curNode),1)
    
    #policy network
    suffix = "Continuous"
    if discreteActionsSpace:
        suffix = "Discrete"
    policyParamsScope = "PolicyNetwork{}".format(suffix)
    with tf.variable_scope(policyParamsScope):
        curNode = tf.layers.Dense(args.hidden_layers_policy[0], activation, kernel_initializer=initializationHidden, name="fc1")(obsPh)
        for i,l in enumerate(args.hidden_layers_policy[1:]):
            curNode = tf.layers.Dense(l, activation, kernel_initializer=initializationHidden, name="fc{}".format(i+2))(curNode)
           
        actionMeanOp = tf.layers.Dense(outputLength, kernel_initializer=initializationFinalPolicy, name="outputA")(curNode)
        actionLogStdOp = tf.get_variable(name="ActionsLogStdDetachedTrainable", initializer=-0.3*np.ones((1, outputLength), dtype=np.float32), trainable=True)
        actionStdOp = tf.math.exp(actionLogStdOp)                
        actionFinalOp = actionMeanOp + tf.random_normal(tf.shape(actionMeanOp)) * actionStdOp 
        sampledLogProbsOp = utils.gaussian_likelihood(actionFinalOp, actionMeanOp, actionLogStdOp)
        logProbWithCurrParamsOp = utils.gaussian_likelihood(aPh, actionMeanOp, actionLogStdOp) 
    
    #definition of losses to optimize
    ratio = tf.exp(logProbWithCurrParamsOp - logProbSampPh)
    Lloss = -tf.reduce_mean(ratio*advPh) # - sign because we want to maximize our objective
    
    if args.plus and args.plus_eps > 0:
        vLossUncliped = (vfOutputOp - totalEstimatedDiscountedRewardPh)**2
        vClipped = VPrevPh + tf.clip_by_value(vfOutputOp - VPrevPh, -args.plus_eps, args.plus_eps)
        vLossClipped = (vClipped - totalEstimatedDiscountedRewardPh)**2
        vLossMax = tf.maximum(vLossClipped, vLossUncliped)
        stateValueLoss = tf.reduce_mean(0.5 * vLossMax)
    else:
        stateValueLoss = tf.reduce_mean((vfOutputOp - totalEstimatedDiscountedRewardPh)**2)
        
    if(discreteActionsSpace):
        KLcontraint = utils.categorical_kl(logProbWithCurrParamsOp, logProbsAllPh) 
    else:
        KLcontraint = utils.diagonal_gaussian_kl(actionMeanOp, actionLogStdOp, oldActionMeanPh, oldActionLogStdPh)     
    
    if args.plus_plus and args.plus_plus_grad_clip >= 0:
        if args.plus_lr_annealing:
            optimizer = tf.train.AdamOptimizer(learning_rate = learningRatePh, epsilon=args.adam_eps)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate = args.learning_rate_state_value, epsilon=args.adam_eps)    
        valGradients, valVaribales = zip(*optimizer.compute_gradients(stateValueLoss))  
        valGradients, _ = tf.clip_by_global_norm(valGradients, args.plus_plus_grad_clip)       
        svfOptimizationStep = optimizer.apply_gradients(zip(valGradients, valVaribales))
    else:
        svfOptimizationStep = tf.train.AdamOptimizer(args.learning_rate_state_value, epsilon=args.adam_eps).minimize(stateValueLoss)
    
    #other ops
    policyParams = utils.get_vars(policyParamsScope)
    getPolicyParams = utils.flat_concat(policyParams)
    setPolicyParams = utils.assign_params_from_flat(policyParamsFlatten, policyParams)
    
    d, HxOp = utils.hesian_vector_product(KLcontraint, policyParams)
    surrogateFlatLoss = utils.flat_grad(Lloss, policyParams)
    
    if args.damping_coef > 0:
        HxOp += args.damping_coef * d
    
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
                epLen=0
                epTotalRew=0
                
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
                nextReturn = lastValue
            else:
                nextNonTerminal = 1.0 - dones[t+1]
                nextValue = predVals[t+1]
                nextReturn = returns[t+1]
                
            delta = rewards[t] + args.gamma * nextValue * nextNonTerminal - predVals[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.lambd * nextNonTerminal * lastgaelam
            returns[t] = rewards[t] + args.gamma * nextNonTerminal * nextReturn
            
        if not args.orig_returns:
            returns = advantages + predVals
         
        if args.norm_adv:
            advantages = utils.normalize(advantages)
 
        #if minimal is set to false, this will be not used, even though it will be passed in feed_dict for optimization step (see how opt. step is defined)
        learningRateVf = utils.annealedNoise(args.learning_rate_state_value, 0, args.epochs, e)
        
        policyUpdateStart = time.time()  
        if args.fisher_fraction < 1:
            selectedForFisherEstimation = np.random.choice(args.epoch_len, int(args.epoch_len*args.fisher_fraction), replace=False)
        else:
            selectedForFisherEstimation = np.arange(args.epoch_len)
            
        if discreteActionsSpace:
            Hx = lambda newDir : sess.run(HxOp, feed_dict={d : newDir, logProbsAllPh : additionalInfos[0][selectedForFisherEstimation], obsPh : obs[selectedForFisherEstimation]})
        else:
            Hx = lambda newDir : sess.run(HxOp, feed_dict={d : newDir, oldActionMeanPh : additionalInfos[0][selectedForFisherEstimation], oldActionLogStdPh : additionalInfos[1][selectedForFisherEstimation], obsPh : obs[selectedForFisherEstimation]})
        
        grad = sess.run(surrogateFlatLoss, feed_dict = { obsPh : obs, aPh: actions, advPh : advantages, logProbSampPh : sampledLogProb})#, logProbsAllPh : allLogProbs})
        cjStart = time.time()
        newDir = utils.conjugate_gradients(Hx, grad, args.cg_iters)    
        cjEnd = time.time()
        
        if not args.plus:
            curDelta = args.delta
        else:
            curDelta = utils.annealedNoise(args.delta,args.delta_final,args.epochs,e)
            
        LlossOld = sess.run(Lloss , feed_dict={obsPh : obs, aPh: actions, advPh : advantages, logProbSampPh : sampledLogProb})#, logProbsAllPh : allLogProbs})#L function inputs: observations, advantages estimated, logProb of sampled action, logProbsOfAllActions              
        coef = np.sqrt(2*curDelta/(np.dot(np.transpose(newDir),Hx(newDir)) + 1e-8))
        
        oldParams = sess.run(getPolicyParams)
        
        if args.wandb_log:
            wandb.config.policy_params = oldParams.shape[0]    
        
        for j in range(args.max_iters_line_search):
            step = args.alpha**j
            
            #check if KL distance is within limits and is Lloss going down
            sess.run(setPolicyParams, feed_dict={policyParamsFlatten : oldParams - coef*newDir*step})
            if discreteActionsSpace:
                kl, LlossNew = sess.run([KLcontraint, Lloss],  feed_dict={obsPh : obs, aPh: actions, advPh : advantages, logProbSampPh : sampledLogProb, logProbsAllPh : additionalInfos[0]})#same as L function inputs plus
            else:
                kl, LlossNew = sess.run([KLcontraint, Lloss],  feed_dict={obsPh : obs, aPh: actions, advPh : advantages, logProbSampPh : sampledLogProb, oldActionMeanPh : additionalInfos[0], oldActionLogStdPh : additionalInfos[1]})#same as L function inputs plus
            
            if (kl <= curDelta and LlossNew <= LlossOld):
                break
            
            if(j == args.max_iters_line_search - 1):
                sess.run(setPolicyParams, feed_dict={policyParamsFlatten : oldParams})
                LlossNew = LlossOld        
                kl = 0 
                print("Line search didn't find step size that satisfies KL constraint")
        
        policyUpdateEnd = time.time()    
        
        print("\tPolicy in epoch {} updated in {}".format(e, policyUpdateEnd-policyUpdateStart))    
                
        svfUpdateStart = time.time()   
        total = args.epoch_len
        for j in range(args.state_value_network_updates):

            perm = np.random.permutation(total)
            start = 0
            while(start < total):    
                end = np.amin([start+args.value_function_batch_size, total])
                if args.plus and args.plus_lr_annealing:
                    sess.run(svfOptimizationStep, feed_dict={obsPh : obs[perm[start:end]], totalEstimatedDiscountedRewardPh : returns[perm[start:end]], VPrevPh : predVals[perm[start:end]], learningRatePh: learningRateVf})
                else:
                    sess.run(svfOptimizationStep, feed_dict={obsPh : obs[perm[start:end]], totalEstimatedDiscountedRewardPh : returns[perm[start:end]], VPrevPh : predVals[perm[start:end]]})
                    
                start = end
        
        svfUpdateEnd = time.time()     
        
        print("\tState value in epoch {} updated in {}".format(e, svfUpdateEnd-svfUpdateStart))    
           
        #after update, calculate new loss
        svLossStart = time.time()
        svfTrainInBatchesThreshold = 5000
        if obs.shape[0] > 2*svfTrainInBatchesThreshold:
            start = 0
            SVLossSum = 0
            while(start < total):    
                end = np.amin([start+2*svfTrainInBatchesThreshold, total])
                avgBatch = sess.run(stateValueLoss, feed_dict={obsPh : obs[start:end], totalEstimatedDiscountedRewardPh : returns[start:end], VPrevPh : predVals[start:end]})
                SVLossSum += avgBatch*(end-start)
                start = end
            SVLoss = SVLossSum/obs.shape[0]
        else:
            SVLoss = sess.run(stateValueLoss, feed_dict={obsPh : obs, totalEstimatedDiscountedRewardPh : returns, VPrevPh : predVals})            
           
        svLossEnd = time.time()      
        
        print("\tState value loss in epoch {} calculated in {}".format(e, svLossEnd-svLossStart))
              
        summarySVm, summarySurrogateDiff, summaryKL = sess.run([SVLossSummary, SurrogateDiffSum, KLSum], feed_dict = {SVLossPh:SVLoss, SurrogateDiffPh:LlossNew - LlossOld, KLPh:kl})
        writer.add_summary(summarySVm, e)
        writer.add_summary(summarySurrogateDiff, e)
        writer.add_summary(summaryKL, e)        
    
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
                  
        if nextDones[0]:
            testRets.append(testRet)
            testRet = 0
        
        testRet += infoss[0]["origRew"]
        osbTest = nextObss.copy()  
        
    meanLatest = np.mean(testRets[1:])
    summaryLatestTestRet = sess.run(epRewLatestMeanTestSum, feed_dict = {epRewTestPh:meanLatest})
    writer.add_summary(summaryLatestTestRet)   
    
    writer.close()
    env.close()
    