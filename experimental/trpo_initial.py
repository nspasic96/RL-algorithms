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
from EnvironmentWrapper import EnvironmentWrapper
from networks import StateValueNetwork, PolicyNetworkDiscrete, PolicyNetworkContinuous
from GAEBuffer import GAEBuffer
from Statistics import Statistics

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

#test parameters
parser.add_argument('--test_every_n_epochs', type=int, default=5,
                   help='after every n episodes agent without noise will be tested')
parser.add_argument('--test_episodes', type=int, default=20,
                   help='when testing, test_episodes will be played and taken for calculting statistics')
parser.add_argument('--render', type=lambda x: (str(x).lower() == 'true'), default=False,
                   help='whether to render agent when it is being tested')
parser.add_argument('--run_statistics', type=lambda x: (str(x).lower() == 'true'), default=True,
                   help='whether to run statistics, necessary if testing agent')

#TRPO specific parameters
parser.add_argument('--delta', type=float, default=0.01,
                   help='max KL distance between two successive distributions at the start of anneailng')
parser.add_argument('--delta_final', type=float, default=0.01,
                   help='max KL distance between two successive distributions at the end of annealing')
parser.add_argument('--state_value_network_updates', type=int, default=10,
                   help="number of updates for state-value network")
parser.add_argument('--damping_coef', type=float, default=0.1,
                   help='damping coef')
parser.add_argument('--cg_iters', type=int, default=10,
                   help='number of iterations in cg algorithm')
parser.add_argument('--max_iters_line_search', type=int, default=10,
                   help="maximum steps to take in line serach")
parser.add_argument('--alpha', type=float, default=0.8,
                   help="defult step size in line serach")
parser.add_argument('--norm_adv', type=lambda x: (str(x).lower() == 'true'), default=True,
                   help="whether to normalize batch of advantages obtained from GAE buffer for policy optimization")

#parameters related to TRPO+
parser.add_argument('--plus', type=lambda x: (str(x).lower() == 'true'), default=False,
                   help='whether to add code optimizations 1-4')
parser.add_argument('--plus_returns', type=lambda x: (str(x).lower() == 'true'), default=True,
                   help='whether to return returns instead of reward for training')
parser.add_argument('--plus_initialization', type=lambda x: (str(x).lower() == 'true'), default=True,
                   help='whether to initialize weights with orthogonal initializer')
parser.add_argument('--plus_eps', type=float, default=0.2,
                   help='epsilon for clipping value function, negative value to turn it off')
parser.add_argument('--plus_grad_clip', type=float, default=1. ,
                   help='gradient l2 norm, negative value to turn it off')

args = parser.parse_args()

dtype = tf.float32
dtypeNp = np.float32

if not args.seed:
    args.seed = int(time.time())
 
graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    
    env = gym.make(args.gym_id) 
    if args.plus and args.plus_returns:
        env = EnvironmentWrapper(env.env, normOb=True, rewardNormalization="returns", clipOb=10., clipRew=10., episodicMeanVarObs=False, episodicMeanVarRew=False, gamma=args.gamma)         
    else:
        env = EnvironmentWrapper(env.env, normOb=False, rewardNormalization=None, clipOb=1000000., clipRew=1000000)    
        
    np.random.seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    tf.set_random_seed(args.seed)

    discreteActionsSpace = utils.is_discrete(env)
    
    inputLength = env.observation_space.shape[0]
    outputLength = env.action_space.n if discreteActionsSpace else env.action_space.shape[0]
    
    #summeries placeholders and summery scalar objects      
    epRewTestPh = tf.placeholder(tf.float32, shape=None, name='episode_test_real_reward_latest_mean_summary')
    epTotalRewPh = tf.placeholder(dtype, shape=None, name='episode_reward_train_summary')
    epLenPh = tf.placeholder(dtype, shape=None, name='episode_length_train_summary')
    SVLossPh = tf.placeholder(dtype, shape=None, name='value_function_loss_summary')
    SurrogateDiffPh = tf.placeholder(dtype, shape=None, name='surrogate_function_value_summary')
    KLPh = tf.placeholder(dtype, shape=None, name='kl_divergence_summary')
    epRewLatestMeanSum = tf.summary.scalar('episode_test_reward_latest_mean', epRewTestPh)
    epTotalRewSum = tf.summary.scalar('episode_reward_train', epTotalRewPh)
    epLenSum = tf.summary.scalar('episode_length_train', epLenPh)
    SVLossSummary = tf.summary.scalar('value_function_loss', SVLossPh)
    SurrogateDiffSum = tf.summary.scalar('surrogate_function_value', SurrogateDiffPh)
    KLSum = tf.summary.scalar('kl_divergence', KLPh)  
      
    #logging details      
    implSuffix = os.path.basename(__file__).rstrip(".py")
    prefix = "plus-" if args.plus else ""
    experimentName = f"{prefix}{args.gym_id}__{implSuffix}__{args.seed}__{int(time.time())}"
    writer = tf.summary.FileWriter(f"runs/{experimentName}", graph = sess.graph)
    
    if args.wandb_log:
        
        cnf = vars(args)
        cnf['action_space_type'] = 'discrete' if discreteActionsSpace else 'continuous'
        cnf['input_length'] = inputLength
        cnf['output_length'] = outputLength
        cnf['exp_name_tb'] = experimentName
        wandb.init(project=args.wandb_projet_name, config=cnf, name=experimentName, tensorboard=True)
    
    #statistics to run
    if args.run_statistics:
        statSizeRew = args.test_episodes

        statistics = {}
        statistics["test_reward"] = Statistics(statSizeRew, 1, "test_reward")
    
    #definition of placeholders
    logProbSampPh = tf.placeholder(dtype = dtype, shape=[None], name="logProbSampled") #log probabiliy of action sampled from sampling distribution (pi_old)
    advPh = tf.placeholder(dtype = dtype, shape=[None], name="advantages") #advantages obtained using GAE-lambda and values obtainet from StateValueNetwork V
    VPrevPh = tf.placeholder(dtype = dtype, shape=[None], name="previousValues") #values for previous iteration returned by StateValueNetwork V
    totalEstimatedDiscountedRewardPh = tf.placeholder(dtype = dtype, shape=[None], name="totalDiscountedReward") #total discounted cumulative reward
    policyParamsFlatten= tf.placeholder(dtype = dtype, shape=[None], name = "policyParams") #policy params flatten, used in assingment of pi params in line search algorithm
    obsPh = tf.placeholder(dtype=dtype, shape=[None, inputLength], name="observations") #observations 
    learningRatePh = tf.placeholder(dtype=dtype, shape=None, name="learningRatePh")#learning rate placeholder, used when TRPO+ is enabled
    
    if discreteActionsSpace:
        aPh = tf.placeholder(dtype=tf.int32, shape=[None,1], name="actions") #actions taken
        logProbsAllPh = tf.placeholder(dtype= dtype, shape=[None, outputLength], name="logProbsAll") #log probabilities of all actions according to sampling distribution (pi_old)
        additionalInfoLengths = [outputLength]
    else:
        aPh = tf.placeholder(dtype=dtype, shape=[None,outputLength], name="actions")
        oldActionMeanPh = tf.placeholder(dtype=dtype, shape=[None,outputLength], name="actionsMeanOld")
        oldActionLogStdPh = tf.placeholder(dtype=dtype, shape=[None,outputLength], name="actionsLogStdOld") 
        additionalInfoLengths = [outputLength, outputLength]
    
    buffer = GAEBuffer(args.gamma, args.lambd, args.epoch_len, inputLength, 1 if discreteActionsSpace else outputLength, args.norm_adv, additionalInfoLengths)
   
    #definition of networks
    orthogonalInitializtionV=[2**0.5]*len(args.hidden_layers_state_value) + [1] if (args.plus and args.plus_initialization) else False
    orthogonalInitializtionP=[2**0.5]*len(args.hidden_layers_policy) + [0.01] if (args.plus and args.plus_initialization) else False
    V = StateValueNetwork(sess, inputLength, args.hidden_layers_state_value, args.learning_rate_state_value, obsPh, orthogonalInitializtion=orthogonalInitializtionV) #this network has method for training, but it is never used. Instead, training is done outside of this class
    if(discreteActionsSpace):
        policy = PolicyNetworkDiscrete(sess, inputLength, outputLength, args.hidden_layers_policy, obsPh, aPh, "Orig", orthogonalInitializtion=orthogonalInitializtionP) #policy network for discrete action space
    else:          
        policyActivations = [tf.nn.tanh for i in range(len(args.hidden_layers_policy))] + [None]    
        policy = PolicyNetworkContinuous(sess, inputLength, outputLength, args.hidden_layers_policy, policyActivations, obsPh, aPh, "Orig", logStdInit=-0.5*np.ones((1, outputLength), dtype=np.float32), logStdTrainable=True, orthogonalInitializtion=orthogonalInitializtionP)
      
    #definition of losses to optimize
    ratio = tf.exp(policy.logProbWithCurrParams - logProbSampPh)
    Lloss = -tf.reduce_mean(ratio*advPh) # - sign because we want to maximize our objective
    
    if  args.plus and args.plus_eps >= 0:
        vLossUncliped = (V.output - totalEstimatedDiscountedRewardPh)**2
        vClipped = VPrevPh + tf.clip_by_value(V.output - VPrevPh, -args.plus_eps, args.plus_eps)
        vLossClipped = (vClipped - totalEstimatedDiscountedRewardPh)**2
        vLossMax = tf.maximum(vLossClipped, vLossUncliped)
        stateValueLoss = tf.reduce_mean(0.5 * vLossMax)
    else:
        stateValueLoss = tf.reduce_mean((V.output - totalEstimatedDiscountedRewardPh)**2)
        
    if(discreteActionsSpace):
        KLcontraint = utils.categorical_kl(policy.logProbs, logProbsAllPh) 
    else:
        KLcontraint = utils.diagonal_gaussian_kl(policy.actionMean, policy.actionLogStd, oldActionMeanPh, oldActionLogStdPh)     
    
    if args.plus and args.plus_grad_clip >= 0:
        optimizer = tf.train.AdamOptimizer(learning_rate = learningRatePh)    
        valGradients, valVaribales = zip(*optimizer.compute_gradients(stateValueLoss))  
        valGradients, _ = tf.clip_by_global_norm(valGradients, args.plus_grad_clip)       
        svfOptimizationStep = optimizer.apply_gradients(zip(valGradients, valVaribales))
    else:
        svfOptimizationStep = tf.train.AdamOptimizer(args.learning_rate_state_value).minimize(stateValueLoss)
    
    #other ops
    suffix = "ContinuousOrig"
    if discreteActionsSpace:
        suffix = "DiscreteOrig"
    policyParams = utils.get_vars("PolicyNetwork"+ suffix)
    getPolicyParams = utils.flat_concat(policyParams)
    setPolicyParams = utils.assign_params_from_flat(policyParamsFlatten, policyParams)
    
    d, HxOp = utils.hesian_vector_product(KLcontraint, policyParams)
    
    if args.damping_coef > 0:
        HxOp += args.damping_coef * d
    
    #tf session initialization
    init = tf.initialize_local_variables()
    init2 = tf.initialize_all_variables()
    sess.run([init,init2])
    
    #algorithm
    finishedEp = 0
    evaluationNum = 0
    for e in range(args.epochs):    
        print("Epoch {} started".format(e))
        obs, epLen, epTotalRew = env.reset().copy(), 0, 0
        
        epochSt = time.time()
        for l in range(args.epoch_len):
            
            if discreteActionsSpace:
                sampledAction, logProbSampledAction, logProbsAll = policy.getSampledActions(np.expand_dims(obs, 0))
                additionalInfos = [logProbsAll]
            else:
                sampledAction, logProbSampledAction, actionsMean, actionLogStd = policy.getSampledActions(np.expand_dims(obs, 0))
                additionalInfos = [actionsMean, actionLogStd]
                
            predictedV = V.forward(np.expand_dims(obs, 0))
            
            nextObs, reward, terminal, infos = env.step(sampledAction[0])  
            epLen += 1
            epTotalRew += infos["origRew"]
            buffer.add(obs, sampledAction[0], predictedV[0][0], logProbSampledAction, reward, additionalInfos)
            obs = nextObs.copy()
    
            done = terminal or epLen == args.max_episode_len
            if(done or l == args.epoch_len -1):
                #if(not terminal):
                    #print("Cutting path. Either max episode length steps are done in current episode or epoch has finished")
                val = 0 if terminal else V.forward(np.expand_dims(obs, 0))
                buffer.finishPath(val)
                if terminal and args.wandb_log:
                    summaryRet, summaryLen = sess.run([epTotalRewSum, epLenSum], feed_dict = {epTotalRewPh:epTotalRew, epLenPh:epLen})
                    writer.add_summary(summaryRet, finishedEp)
                    writer.add_summary(summaryLen, finishedEp)                    
                    finishedEp += 1                
                obs, epLen, epTotalRew = env.reset().copy(), 0, 0
                
        simulationEnd = time.time()      
        
        print("\tSimulation in epoch {} finished in {}".format(e, simulationEnd-epochSt))  
            
        #update policy and update state-value(multiple times) after that
        observations, actions, advEst, sampledLogProb, returns, Vprevs, additionalInfos = buffer.get()
            
        policyUpdateStart = time.time()  
        if discreteActionsSpace:
            Hx = lambda newDir : sess.run(HxOp, feed_dict={d : newDir, logProbsAllPh : additionalInfos[0], obsPh : observations})
        else:
            Hx = lambda newDir : sess.run(HxOp, feed_dict={d : newDir, oldActionMeanPh : additionalInfos[0], oldActionLogStdPh : additionalInfos[1], obsPh : observations})

        grad = sess.run(utils.flat_grad(Lloss, policyParams), feed_dict = { obsPh : observations, aPh: actions, advPh : advEst, logProbSampPh : sampledLogProb})#, logProbsAllPh : allLogProbs})
        cjStart = time.time()
        newDir = utils.conjugate_gradients(Hx, grad, args.cg_iters)    
        cjEnd = time.time()
        
        curDelta = utils.annealedNoise(args.delta,args.delta_final,args.epochs,e)
        LlossOld = sess.run(Lloss , feed_dict={obsPh : observations, aPh: actions, advPh : advEst, logProbSampPh : sampledLogProb})#, logProbsAllPh : allLogProbs})#L function inputs: observations, advantages estimated, logProb of sampled action, logProbsOfAllActions              
        coef = np.sqrt(2*curDelta/(np.dot(np.transpose(newDir),Hx(newDir)) + 1e-8))
        
        oldParams = sess.run(getPolicyParams)
        
        if args.wandb_log:
            wandb.config.policy_params = oldParams.shape[0]    
        
        for j in range(args.max_iters_line_search):
            step = args.alpha**j
            
            #check if KL distance is within limits and is Lloss going down
            sess.run(setPolicyParams, feed_dict={policyParamsFlatten : oldParams - coef*newDir*step})
            if discreteActionsSpace:
                kl, LlossNew = sess.run([KLcontraint, Lloss],  feed_dict={obsPh : observations, aPh: actions, advPh : advEst, logProbSampPh : sampledLogProb, logProbsAllPh : additionalInfos[0]})#same as L function inputs plus
            else:
                kl, LlossNew = sess.run([KLcontraint, Lloss],  feed_dict={obsPh : observations, aPh: actions, advPh : advEst, logProbSampPh : sampledLogProb, oldActionMeanPh : additionalInfos[0], oldActionLogStdPh : additionalInfos[1]})#same as L function inputs plus
            
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
                if args.plus:
                    sess.run(svfOptimizationStep, feed_dict={obsPh : observations[perm[start:end]], totalEstimatedDiscountedRewardPh : returns[perm[start:end]], VPrevPh : Vprevs[perm[start:end]], learningRatePh: utils.annealedNoise(args.learning_rate_state_value,0,args.epochs,e)})
                else:
                    sess.run(svfOptimizationStep, feed_dict={obsPh : observations[perm[start:end]], totalEstimatedDiscountedRewardPh : returns[perm[start:end]], VPrevPh : Vprevs[perm[start:end]]})
                    
                start = end
        
        svfUpdateEnd = time.time()     
        
        print("\tState value in epoch {} updated in {}".format(e, svfUpdateEnd-svfUpdateStart))    
           
        #after update, calculate new loss
        svLossStart = time.time()
        svfTrainInBatchesThreshold = 5000
        if observations.shape[0] > 2*svfTrainInBatchesThreshold:
            start = 0
            SVLossSum = 0
            while(start < total):    
                end = np.amin([start+2*svfTrainInBatchesThreshold, total])
                avgBatch = sess.run(stateValueLoss, feed_dict={obsPh : observations[start:end], totalEstimatedDiscountedRewardPh : returns[start:end], VPrevPh : Vprevs[start:end]})
                SVLossSum += avgBatch*(end-start)
                start = end
            SVLoss = SVLossSum/observations.shape[0]
        else:
            SVLoss = sess.run(stateValueLoss, feed_dict={obsPh : observations, totalEstimatedDiscountedRewardPh : returns, VPrevPh : Vprevs})            
           
        svLossEnd = time.time()      
        
        print("\tState value loss in epoch {} calculated in {}".format(e, svLossEnd-svLossStart))
              
        summarySVm, summarySurrogateDiff, summaryKL = sess.run([SVLossSummary, SurrogateDiffSum, KLSum], feed_dict = {SVLossPh:SVLoss, SurrogateDiffPh:LlossNew - LlossOld, KLPh:kl})
        writer.add_summary(summarySVm, e)
        writer.add_summary(summarySurrogateDiff, e)
        writer.add_summary(summaryKL, e)        
    
        epochEnd = time.time()
        print("Epoch {} ended in {}".format(e, epochEnd-epochSt))
        
        if(e % args.test_every_n_epochs == 0):
            evaluationNum += 1
            print("Testing agent without noise for {} episodes after {} epochs".format(args.test_episodes, e))
            for _ in range(args.test_episodes):
                osbTest = env.reset()
                testRet = 0
                for _ in range(args.max_episode_len):
                    if args.render:
                        env.render()
                    _, _, sampledActionTest, _ = policy.getSampledActions(np.expand_dims(osbTest, 0))  
                    nextOsbTest, _, terminalTest, infosTest = env.step(sampledActionTest[0])
                    testRet += infosTest["origRew"]
                    osbTest = nextOsbTest
                    if terminalTest:
                        break  
                statistics["test_reward"].addValue(np.asarray([[testRet]]))
                
            meanLatest = statistics["test_reward"].getMeans()[0]
            summaryLatestRet = sess.run(epRewLatestMeanSum, feed_dict = {epRewTestPh:meanLatest})
            writer.add_summary(summaryLatestRet, evaluationNum)      
        
    
    writer.close()
    env.close()
    