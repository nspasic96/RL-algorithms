import argparse
import gym
import pybulletgym
import numpy as np
import tensorflow as tf
import time
import wandb
from collections import deque

import sys
sys.path.append("../")

import utils
from networks import QNetwork, PolicyNetworkContinuous
from ReplayBuffer import ReplayBuffer
from Statistics import Statistics
from EnvironmentWrapper import EnvironmentWrapper

parser = argparse.ArgumentParser(description='TD3')

parser.add_argument('--gym-id', type=str, default="HopperPyBulletEnv-v0",
                   help='the id of the gym environment')
parser.add_argument('--seed', type=int,
                   help='seed of the experiment')
parser.add_argument('--total_train_steps', type=int, default=1000000,
                   help="total number of steps to train")
parser.add_argument('--gamma', type=float, default=0.99,
                   help='the discount factor gamma')
parser.add_argument('--rho', type=float, default=0.005,
                   help='polyak')
parser.add_argument('--learning_rate_q', type=float, default=1e-3,
                   help='learning rate of the optimizer of q function')
parser.add_argument('--hidden_layers_q', type=int, nargs='+', default=[400,300],
                   help='hidden layers size in state value network')
parser.add_argument('--learning_rate_policy', type=float, default=1e-3,
                   help='learning rate of the optimizer of policy function')
parser.add_argument('--hidden_layers_policy', type=int, nargs='+', default=[400,300],
                   help='hidden layers size in policy network')
parser.add_argument('--buffer_size', type=int, default=-1,
                   help='size of reply buffer, paper sugests this to be equal to number of steps')
parser.add_argument('--batch_size', type=int, default=100,
                   help='number of samples from reply buffer to train on')
parser.add_argument('--update_after', type=int, default=10000,
                   help='number of samples in buffer before first update')
parser.add_argument('--update_freq', type=int, default=2,
                   help='update networks every update_freq steps')
parser.add_argument('--start_steps', type=int, default=10000,
                   help='steps to perform random policy at start')
parser.add_argument('--eps_start', type=float, default=0.1,
                   help='start gaussian noise to add to mean action value')
parser.add_argument('--eps_end', type=float, default=0.1,
                   help='end gaussian noise to add to mean action value')
parser.add_argument('--sigma_hat', type=float, default=0.2,
                   help='sigma for noise to add to target policy generated actions when updating critics')
parser.add_argument('--target_action_noise_clip', type=float, default=0.5,
                   help='clip noise from previous parameter')
parser.add_argument('--steps_to_decrease', type=int, default=100000,
                   help='steps to anneal noise from start to end')
parser.add_argument('--max_episode_len', type=int, default=1000,
                   help="max length of one episode")
parser.add_argument('--wandb_projet_name', type=str, default="twin-delayed-deep-deterministic",
                   help="the wandb's project name")
parser.add_argument('--wandb_log', type=bool, default=False,
                   help='whether to log results to wandb')
parser.add_argument('--run_statistics', type=bool, default=True,
                   help='whether to log histograms for actions and observations')
parser.add_argument('--norm_obs', type=bool, default=False,
                   help='whether to normalize observations')
parser.add_argument('--norm_rew', type=bool, default=False,
                   help='whether to normalize rewards')
parser.add_argument('--clip_obs', type=float, default=1000000,
                   help='observations clipping')
parser.add_argument('--clip_rew', type=float, default=1000000,
                   help='reward clipping')
parser.add_argument('--test_every_n_steps', type=int, default=5000,
                   help='every nth step will triger evaluation, set to total_train_steps not to evaluate at all')
parser.add_argument('--test_episodes', type=int, default=10,
                   help='test agent on more than one episode for more accurate approximation')
parser.add_argument('--render', type=bool, default=False,
                   help='whether to render when testing')
args = parser.parse_args()

if not args.seed:
    args.seed = int(time.time())

if args.buffer_size == -1:
    args.buffer_size = args.total_train_steps
    print("\nBuffer size not specified. Taking value of {} which is the same as total_train_steps, as suggested by the paper\n".format(args.buffer_size)) 

graph = tf.Graph()
with tf.Session(graph=graph) as sess:

    env = gym.make(args.gym_id)  
    env = EnvironmentWrapper(env.env, args.norm_obs, args.norm_rew, args.clip_obs, args.clip_rew)  
    np.random.seed(args.seed)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    tf.set_random_seed(args.seed)
    
    if utils.is_discrete(env):
        exit("TD3 can only be applied to continuous action space environments")
    
    inputLength = env.observation_space.shape[0]
    outputLength = env.action_space.shape[0]
    
    #summeries placeholders and summery scalar objects   
    epRewPh = tf.placeholder(tf.float32, shape=None, name='episode_reward_summary')
    epRewLatestMeanPh = tf.placeholder(tf.float32, shape=None, name='episode_reward_latest_mean_summary')
    epLenPh = tf.placeholder(tf.float32, shape=None, name='episode_length_summary')
    expVarPh = tf.placeholder(tf.float32, shape=None, name='explained_variance_summary')
    QLossPh = tf.placeholder(tf.float32, shape=None, name='q_function_loss_summary')
    policyLossPh = tf.placeholder(tf.float32, shape=None, name='policy_function_value_summary')
    epRewSum = tf.summary.scalar('episode_reward', epRewPh)
    epRewLatestMeanSum = tf.summary.scalar('episode_reward_latest_mean', epRewLatestMeanPh)
    epLenSum = tf.summary.scalar('episode_length', epLenPh)
    expVarSum = tf.summary.scalar('explained_variance', expVarPh)
    QLossSum = tf.summary.scalar('Losses/q_function_loss', QLossPh)
    policyLossSum = tf.summary.scalar('Losses/policy_function_loss', policyLossPh)  
            
    implSuffix = "experimental"
    experimentName = f"{args.gym_id}__td3_{implSuffix}__{args.seed}__{int(time.time())}"
    writer = tf.summary.FileWriter(f"runs/{experimentName}", graph = sess.graph)

    if args.wandb_log:
        cnf = vars(args)
        cnf['action_space_type'] = 'continuous'
        cnf['input_length'] = inputLength
        cnf['reward_threshold'] = env.spec.reward_threshold
        cnf['output_length'] = outputLength
        wandb.init(project=args.wandb_projet_name, config=cnf, name=experimentName, tensorboard=True)
    
    #definition of placeholders
    rewPh = tf.placeholder(dtype = tf.float32, shape=[None], name="rewards") 
    terPh = tf.placeholder(dtype = tf.float32, shape=[None], name="terminals")
    obsPh = tf.placeholder(dtype=tf.float32, shape=[None, inputLength], name="observations")   
    nextObsPh = tf.placeholder(dtype=tf.float32, shape=[None, inputLength], name="nextObservations")  
    aPh = tf.placeholder(dtype=tf.float32, shape=[None,outputLength], name="actions")
    noise = tf.clip_by_value(tf.random.normal([1,outputLength], 0, args.sigma_hat), -args.target_action_noise_clip, args.target_action_noise_clip)
    aWithNoise = aPh + noise
    
    #definition of networks
    clip = np.zeros(shape=(2,outputLength))
    clip[0,:] = env.action_space.low
    clip[1,:] = env.action_space.high
    policyActivations = [tf.nn.relu for i in range(len(args.hidden_layers_policy))] + [tf.nn.tanh]
    qActivations = [tf.nn.relu for i in range(len(args.hidden_layers_q))] + [None]

    hiddenLayerMergeWithAction = 0
    policy = PolicyNetworkContinuous(sess, inputLength, outputLength, args.hidden_layers_policy, policyActivations, obsPh, aPh, "Orig", actionMeanScale=np.expand_dims(clip[1,:],0), logStdInit=None, logStdTrainable=False, actionClip=clip)
    policyTarget = PolicyNetworkContinuous(sess, inputLength, outputLength, args.hidden_layers_policy, policyActivations, nextObsPh, aPh, "Target", actionMeanScale=np.expand_dims(clip[1,:],0), logStdInit=None, logStdTrainable=False, actionClip=clip)
    Q1 = QNetwork(sess, inputLength, outputLength, args.hidden_layers_q, qActivations, obsPh, aPh, hiddenLayerMergeWithAction, suffix="Orig1") # original Q network 1 
    Q2 = QNetwork(sess, inputLength, outputLength, args.hidden_layers_q, qActivations, obsPh, aPh, hiddenLayerMergeWithAction, suffix="Orig2") # original Q network 2 
    QAux1 = QNetwork(sess, inputLength, outputLength, args.hidden_layers_q, qActivations, obsPh,  policy.actionFinal, hiddenLayerMergeWithAction, suffix="Aux1", reuse=Q1) # network with parameters same as original Q1 network, but instead of action placeholder it takse output from current policy
    QTarget1 = QNetwork(sess, inputLength, outputLength, args.hidden_layers_q, qActivations, nextObsPh, aWithNoise, hiddenLayerMergeWithAction, suffix="Target1") #target Q1 network, instead of action placeholder it takse output from target policy
    QTarget2 = QNetwork(sess, inputLength, outputLength, args.hidden_layers_q, qActivations, nextObsPh, aWithNoise, hiddenLayerMergeWithAction, suffix="Target2") #target Q2 network, instead of action placeholder it takse output from target policy

    #definition of losses to optimize
    policyLoss = -tf.reduce_mean(QAux1.output)# - sign because we want to maximize our objective    
    targets = tf.stop_gradient(rewPh + args.gamma*(1-terPh)*tf.math.minimum(QTarget1.output, QTarget2.output))
    q1Loss = tf.reduce_mean((Q1.output - targets)**2)
    q2Loss = tf.reduce_mean((Q2.output - targets)**2)
    
    q1Params = utils.get_vars(Q1.variablesScope)
    q2Params = utils.get_vars(Q2.variablesScope)
    q1OptimizationStep = tf.train.AdamOptimizer(learning_rate = args.learning_rate_q).minimize(q1Loss, var_list = q1Params)
    q2OptimizationStep = tf.train.AdamOptimizer(learning_rate = args.learning_rate_q).minimize(q2Loss, var_list = q2Params)
    policyParams = utils.get_vars(policy.variablesScope)
    policyOptimizationStep = tf.train.AdamOptimizer(learning_rate = args.learning_rate_policy).minimize(policyLoss, var_list=policyParams)

    #tf session initialization
    init = tf.initialize_local_variables()
    init2 = tf.initialize_all_variables()
    sess.run([init,init2])
    
    #algorithm
    finishedEp = 0
    evaluationNum = 0
    step = 0
    updates=0
    buffer = ReplayBuffer(args.buffer_size)

    if args.run_statistics:
        statSizeActObs = 10000
        statSizeRew = args.test_episodes
        statSizeExpVar = 10000

        statistics = []
        statistics.append(Statistics(statSizeActObs, inputLength, "observation", True))
        statistics.append(Statistics(statSizeActObs, outputLength, "action", True))
        statistics.append(Statistics(statSizeRew, 1, "rewards"))
        statistics.append(Statistics(statSizeExpVar, 2, "explained_variance"))

    #sync target and 'normal' network
    sess.run(utils.polyak(QTarget1, Q1, 1, sess, False))
    sess.run(utils.polyak(QTarget2, Q2, 1, sess, False))
    sess.run(utils.polyak(policyTarget, policy, 1, sess, False))

    #get target update ops  
    Q1TargetUpdateOp = utils.polyak(QTarget1, Q1, args.rho, sess, verbose=False)
    Q2TargetUpdateOp = utils.polyak(QTarget2, Q2, args.rho, sess, verbose=False)
    policyTargetUpdateOp = utils.polyak(policyTarget, policy, args.rho, sess, verbose=False)

    solved = False
    while step < args.total_train_steps and not solved:  

        episodeStart = time.time()

        obs, epLen, epRet, allRets, allQs, doSample = env.reset(), 0, 0, [], [], True

        #basicaly this is one episode because while exits when terminal state is reached or max number of steps(in episode or generaly) is reached
        while doSample: 

            if step < args.start_steps:    
                sampledAction = np.asarray([env.action_space.sample()])
            else:
                noise = utils.annealedNoise(args.eps_start, args.eps_end, args.steps_to_decrease, step)
                sampledAction, _, = policy.getSampledActions(np.expand_dims(obs, 0)) + np.random.normal(0, noise,(1,outputLength))

            statistics[0].addValue(np.expand_dims(obs, 0))
            statistics[1].addValue(sampledAction)
           
            predQ = sess.run(Q1.output, feed_dict={obsPh:np.expand_dims(obs, 0), aPh:sampledAction})[0]

            nextObs, reward, terminal, infos = env.step(sampledAction[0])
            epLen += 1
            epRet += infos["origRew"] if args.norm_rew else reward
            allRets.append(reward)
            allQs.append(predQ)

            buffer.add(np.expand_dims(obs, 0), sampledAction, reward, np.expand_dims(nextObs, 0), terminal)
            obs = nextObs

            doSample = not terminal and epLen < args.max_episode_len and step < args.total_train_steps
            
            step +=1  

            if terminal or epLen == args.max_episode_len : 
                finishedEp += 1
                for i in range(len(allRets)-2,-1,-1):
                    allRets[i] += args.gamma*allRets[i+1]
                    statistics[3].addValue(np.asarray([[allRets[i], allQs[i]]]))
                
                explainedVar = statistics[3].getExplainedVariance()

                summaryRet, summaryLen, summaryExpVar = sess.run([epRewSum, epLenSum, expVarSum], feed_dict = {epRewPh:epRet, epLenPh:epLen, expVarPh:explainedVar})

                writer.add_summary(summaryRet, finishedEp)
                writer.add_summary(summaryLen, finishedEp)  
                writer.add_summary(summaryExpVar, finishedEp)

            #test deterministic agent
            if(step % args.test_every_n_steps == 0):
                evaluationNum += 1
                print("Testing agent with no exploration noise for {} episodes".format(args.test_episodes))
                for _ in range(args.test_episodes):
                    osbTest = env.reset()
                    testRet = 0
                    for _ in range(args.max_episode_len):
                        if args.render:
                            env.render()
                        sampledActionTest, _ = policy.getSampledActions(np.expand_dims(osbTest, 0))  
                        nextOsbTest, reward, terminalTest, _ = env.step(sampledActionTest[0])
                        testRet += reward
                        osbTest = nextOsbTest
                        if terminalTest:
                            break  
                    statistics[2].addValue(np.asarray([[testRet]]))
                    
                meanLatest = statistics[2].getMeans()[0]
                summaryLatestRet = sess.run(epRewLatestMeanSum, feed_dict = {epRewLatestMeanPh:meanLatest})
                writer.add_summary(summaryLatestRet, evaluationNum)  

                if env.spec.reward_threshold is not None and env.spec.reward_threshold != 0:
                    if(env.spec.reward_threshold > 0):
                        th = env.spec.reward_threshold*0.95
                    else:
                        th = env.spec.reward_threshold/0.95
                    if th < meanLatest:
                        print("Environment solved in step after {} episodes. Variance of reward is {}% of the mean".format(finishedEp, meanLatest/statistics[2].getVars()[0]))
                        solved = True

                    
            #update critics every step
            if buffer.curr >= args.batch_size:
                observations, actions, rewards, nextObservations, terminals = buffer.sample(args.batch_size) 
                sess.run([q1OptimizationStep, q2OptimizationStep], feed_dict={obsPh:observations, nextObsPh:nextObservations, rewPh:rewards, terPh:terminals, aPh:actions})
                    
            #time for update
            if step > args.update_after and step % args.update_freq == 0:   

                for _ in range(args.update_freq):        
                    sess.run(policyOptimizationStep, feed_dict={obsPh:observations})

                    qLossNew, policyLossNew = sess.run([q1Loss, policyLoss], feed_dict={obsPh:observations, nextObsPh:nextObservations, rewPh:rewards, terPh:terminals, aPh:actions})
                    
                    summaryQ, summaryP = sess.run([QLossSum, policyLossSum], feed_dict = {QLossPh:qLossNew, policyLossPh:policyLossNew})
                    writer.add_summary(summaryQ, updates)
                    writer.add_summary(summaryP, updates)  
                    updates +=1
                    
                    sess.run(Q1TargetUpdateOp)
                    sess.run(Q2TargetUpdateOp)
                    sess.run(policyTargetUpdateOp)
        
        episodeEnd = time.time()

        if args.run_statistics:
            feedD = {}
            total =0
            histSummaries = []
            for hist in statistics[:2]:
                values = hist.getValues()
                for i in range(hist.dimensions):
                    feedD[hist.phs[i]] = values[i]
                total += hist.dimensions
                histSummaries.extend(hist.summaries)

            summEval = sess.run(histSummaries, feed_dict = feedD)
            for i in range(total):
                writer.add_summary(summEval[i], global_step=finishedEp)

        print("Episode {} took {}s for {} steps".format(finishedEp , episodeEnd - episodeStart, epLen))
    
    writer.close()
    env.close()