import argparse
import gym
import pybulletgym
import numpy as np
import tensorflow as tf
import time
import wandb

import utils
from networks import QNetwork, PolicyNetworkDiscrete, PolicyNetworkContinuous
from ReplayBuffer import ReplayBuffer

parser = argparse.ArgumentParser(description='DDPG')

parser.add_argument('--gym-id', type=str, default="Pong-ram-v0",
                   help='the id of the gym environment')
parser.add_argument('--seed', type=int, default=1,
                   help='seed of the experiment')
parser.add_argument('--epochs', type=int, default=1000,
                   help="epochs to train")
parser.add_argument('--epoch_len', type=int, default=4000,
                   help="length of one epoch")
parser.add_argument('--gamma', type=float, default=0.99,
                   help='the discount factor gamma')
parser.add_argument('--delta', type=float, default=0.01,
                   help='max KL distance between two successive distributions')
parser.add_argument('--learning_rate_state_value', type=float, default=1e-3,
                   help='learning rate of the optimizer of state-value function')
parser.add_argument('--hidden_layers_state_value', type=int, nargs='+', default=[64,64],
                   help='hidden layers size in state value network')
parser.add_argument('--hidden_layers_policy', type=int, nargs='+', default=[64,64],
                   help='hidden layers size in policy network')
parser.add_argument('--state_value_network_updates', type=int, default=80,
                   help="number of updates for state-value network")
parser.add_argument('--damping_coef', type=float, default=0.1,
                   help='damping coef')
parser.add_argument('--cg_iters', type=int, default=10,
                   help='number of iterations in cg algorithm')
parser.add_argument('--max_iters_line_search', type=int, default=10,
                   help="maximum steps to take in line serach")
parser.add_argument('--alpha', type=float, default=0.8,
                   help="defult step size in line serach")
parser.add_argument('--lambd', type=float, default=0.97,
                   help='lambda for GAE-Lambda')
parser.add_argument('--max_episode_len', type=int, default=1000,
                   help="max length of one episode")
parser.add_argument('--wandb_projet_name', type=str, default="trust-region-policy-optimization",
                   help="the wandb's project name")
parser.add_argument('--exp_name', type=str, default=None,
                   help='the name of this experiment')
parser.add_argument('--play_every_nth_epoch', type=int, default=10,
                   help='every nth epoch will be rendered, set to epochs+1 not to render at all')
args = parser.parse_args()

svfTrainInBatchesThreshold = 5000

if not args.seed:
    args.seed = int(time.time())
    
graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    env = gym.make(args.gym_id)
    if utils.is_discrete(env):
        exit("DDPG can only be applied to continuous action space environments")
    
    inputLength = env.observation_space.shape[0]
    outputLength = env.action_space.shape[0]
    
    #summeries placeholders and summery scalar objects
    epRewPh = tf.placeholder(tf.float32, shape=None, name='episode_reward_summary')
    epLenPh = tf.placeholder(tf.float32, shape=None, name='episode_length_summary')
    SVLossPh = tf.placeholder(tf.float32, shape=None, name='value_function_loss_summary')
    SurrogateDiffPh = tf.placeholder(tf.float32, shape=None, name='surrogate_function_value_summary')
    KLPh = tf.placeholder(tf.float32, shape=None, name='kl_divergence_summary')
    epRewSum = tf.summary.scalar('episode_reward', epRewPh)
    epLenSum = tf.summary.scalar('episode_length', epLenPh)
    SVLossSummary = tf.summary.scalar('value_function_loss', SVLossPh)
    SurrogateDiffSum = tf.summary.scalar('surrogate_function_value', SurrogateDiffPh)
    KLSum = tf.summary.scalar('kl_divergence', KLPh)  
            
    if args.exp_name is not None:
        
        experimentName = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        writer = tf.summary.FileWriter(f"runs/{experimentName}", graph = sess.graph)
        cnf = vars(args)
        cnf['action_space_type'] = 'continuous'
        cnf['input_length'] = inputLength
        cnf['output_length'] = outputLength
        cnf['exp_name_tb'] = experimentName
        note = ""
        if(args.epoch_len > svfTrainInBatchesThreshold):
            note = "State value training split in batches of size {} because of too large number of samples".format(svfTrainInBatchesThreshold)
        wandb.init(project=args.wandb_projet_name, config=cnf, name=args.exp_name, notes=note, tensorboard=True)
    
    #definition of placeholders
    logProbSampPh = tf.placeholder(dtype = tf.float32, shape=[None], name="logProbSampled") #log probabiliy of action sampled from sampling distribution (pi_old)
    rewPh = tf.placeholder(dtype = tf.float32, shape=[None], name="rewards") #advantages obtained using GAE-lambda and values obtainet from QNetwork Q
    terPh = tf.placeholder(dtype = tf.float32, shape=[None], name="terminals") #advantages obtained using GAE-lambda and values obtainet from QNetwork Q
    returnsPh = tf.placeholder(dtype = tf.float32, shape=[None], name="returns") #total discounted cumulative reward
    policyParamsFlatten= tf.placeholder(dtype = tf.float32, shape=[None], name = "policyParams") #policy params flatten, used in assingment of pi params in line search algorithm
    obsPh = tf.placeholder(dtype=tf.float32, shape=[None, inputLength], name="observations") #observations    
    nextObsPh = tf.placeholder(dtype=tf.float32, shape=[None, inputLength], name="nextObservations") #next observations    
    aPh = tf.placeholder(dtype=tf.float32, shape=[None,outputLength], name="actions")
    oldActionMeanPh = tf.placeholder(dtype=tf.float32, shape=[None,outputLength], name="actionsMeanOld")
    oldActionLogStdPh = tf.placeholder(dtype=tf.float32, shape=[None,outputLength], name="actionsLogStdOld") 
    additionalInfoLengths = [outputLength, outputLength]
    
    buffer = ReplayBuffer()
   
    #definition of networks
    policy = PolicyNetworkContinuous(sess, inputLength, outputLength, args.hidden_layers_policy, obsPh, aPh)
    policyTarget = PolicyNetworkContinuous(sess, inputLength, outputLength, args.hidden_layers_policy, nextObsPh, aPh)
    Q = QNetwork(sess, inputLength, outputLength, args.hidden_layers_state_value, obsPh, aPh) # original Q network
    QAux = QNetwork(sess, inputLength, outputLength, args.hidden_layers_state_value, obsPh, policy.actionFinal) # network with parameters same as original Q network, but instead of action placeholder it takse output from current policy
    QTarget = QNetwork(sess, inputLength, outputLength, args.hidden_layers_state_value, nextObsPh, policyTarget.actionFinal) #target Q network, instead of action placeholder it takse output from target policy

    #definition of losses to optimize
    policyLoss = -tf.reduce_mean(QAux.output)# - sign because we want to maximize our objective    
    targets = tf.stop_gradient(rewPh + args.gamma*(1-terPh)*QTarget.output)#check dimensions
    qLoss = tf.reduce_mean((Q.output - targets)**2)
    
    qOptimizationStep = tf.train.AdamOptimizer(learning_rate = args.learning_rate_q).minimize(qLoss)
    policyParams = utils.get_vars("PolicyNetworkContinuous")
    policyOptimizationStep = tf.train.AdamOptimizer(learning_rate = args.learning_rate_policy).minimize(policyLoss, var_list=policyParams)

    #tf session initialization
    init = tf.initialize_local_variables()
    init2 = tf.initialize_all_variables()
    sess.run([init,init2])
    
    #algorithm
    finishedEp = 0
    step = 0

    while step < args.total_train_steps:  

        obs, epLen, epRet, doSample = env.reset(), 0, 0, True

        #basicaly this is one episode because while exits when terminal state is reached or max number of steps(in episode or generaly) is reached
        while doSample:              
            
            sampledAction, logProbSampledAction, actionsMean, actionLogStd = policy.getSampledActions(np.expand_dims(obs, 0))
            additionalInfos = [actionsMean, actionLogStd]
                
            predictedV = Q.forward(np.expand_dims(obs, 0))
            
            nextObs, reward, terminal, _ = env.step(sampledAction[0])  
            epLen += 1
            epRet += reward

            buffer.add(obs, sampledAction[0], reward, nextObs, terminal)
            obs = nextObs

            doSample = not terminal and epLen < args.max_episode_len and step < args.total_train_steps
            
            if terminal and args.exp_name is not None:
                summaryRet, summaryLen = sess.run([epRewSum, epLenSum], feed_dict = {epRewPh:epRet, epLenPh:epLen})
                writer.add_summary(summaryRet, finishedEp)
                writer.add_summary(summaryLen, finishedEp)                    
                finishedEp += 1
            
            step +=1          
            
            #time for update
            if step % args.update_freq == 0:           
                observations, actions, rewards, nextObservations, terminals= buffer.get()                

                sess.run(qOptimizationStep, feed_dict={obsPh:observations, nextObsPh:nextObservations, rewPh:rewards, terPh:terminals, aPh:actions})
                sess.run(policyOptimizationStep, feed_dict={obsPh:observations})

                #TODO: polyak        
        
        if args.exp_name is not None:       
            summarySVm, summarySurrogateDiff, summaryKL = sess.run([SVLossSummary, SurrogateDiffSum, KLSum], feed_dict = {SVLossPh:SVLoss, SurrogateDiffPh:LlossNew - LlossOld, KLPh:kl})
            writer.add_summary(summarySVm, e)
            writer.add_summary(summarySurrogateDiff, e)
            writer.add_summary(summaryKL, e)        
        
    
    writer.close()
    env.close()
    