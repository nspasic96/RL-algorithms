import argparse
import gym
import pybulletgym
import numpy as np
import tensorflow as tf
import time
import wandb

import utils
from networks import QNetwork, PolicyNetworkContinuous
from ReplayBuffer import ReplayBuffer

parser = argparse.ArgumentParser(description='DDPG')

parser.add_argument('--gym-id', type=str, default="HopperPyBulletEnv-v0",
                   help='the id of the gym environment')
parser.add_argument('--seed', type=int, default=1,
                   help='seed of the experiment')
parser.add_argument('--total_train_steps', type=int, default=1000000,
                   help="total number of steps to train")
parser.add_argument('--gamma', type=float, default=0.99,
                   help='the discount factor gamma')
parser.add_argument('--rho', type=float, default=0.01,
                   help='polyak')
parser.add_argument('--learning_rate_q', type=float, default=1e-3,
                   help='learning rate of the optimizer of q function')
parser.add_argument('--hidden_layers_q', type=int, nargs='+', default=[64,64],
                   help='hidden layers size in state value network')
parser.add_argument('--learning_rate_policy', type=float, default=1e-3,
                   help='learning rate of the optimizer of policy function')
parser.add_argument('--hidden_layers_policy', type=int, nargs='+', default=[64,64],
                   help='hidden layers size in policy network')
parser.add_argument('--buffer_size', type=int, default=100000,
                   help='size of reply buffer')
parser.add_argument('--batch_size', type=int, default=256,
                   help='number of samples from reply buffer to train on')
parser.add_argument('--update_after', type=int, default=100000,
                   help='number of samples in buffer before first update')
parser.add_argument('--update_freq', type=int, default=10000,
                   help='update networks every update_freq steps')
parser.add_argument('--eps', type=float, default=0.1,
                   help='non trainable gaussian noise to add to mean action value')
parser.add_argument('--max_episode_len', type=int, default=1000,
                   help="max length of one episode")
parser.add_argument('--wandb_projet_name', type=str, default="deep-deterministic-policy-gradients",
                   help="the wandb's project name")
parser.add_argument('--exp_name', type=str, default=None,
                   help='the name of this experiment')
parser.add_argument('--play_every_nth_epoch', type=int, default=10,
                   help='every nth epoch will be rendered, set to epochs+1 not to render at all')
args = parser.parse_args()

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
    epRewSum = tf.summary.scalar('episode_reward', epRewPh)
    epLenSum = tf.summary.scalar('episode_length', epLenPh)
            
    if args.exp_name is not None:
        
        experimentName = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        writer = tf.summary.FileWriter(f"runs/{experimentName}", graph = sess.graph)
        cnf = vars(args)
        cnf['action_space_type'] = 'continuous'
        cnf['input_length'] = inputLength
        cnf['output_length'] = outputLength
        cnf['exp_name_tb'] = experimentName
        wandb.init(project=args.wandb_projet_name, config=cnf, name=args.exp_name, tensorboard=True)
    
    #definition of placeholders
    rewPh = tf.placeholder(dtype = tf.float32, shape=[None], name="rewards") 
    terPh = tf.placeholder(dtype = tf.float32, shape=[None], name="terminals")
    obsPh = tf.placeholder(dtype=tf.float32, shape=[None, inputLength], name="observations") #observations    
    nextObsPh = tf.placeholder(dtype=tf.float32, shape=[None, inputLength], name="nextObservations") #next observations    
    aPh = tf.placeholder(dtype=tf.float32, shape=[None,outputLength], name="actions")
    
    #definition of networks
    clip = np.zeros(shape=(2,outputLength))
    clip[0,:] = env.action_space.low
    clip[1,:] = env.action_space.high
    logStdInit = args.eps*np.ones(shape=(1,outputLength), dtype=np.float32)

    policy = PolicyNetworkContinuous(sess, inputLength, outputLength, args.hidden_layers_policy, obsPh, aPh, "Orig", logStdInit=logStdInit, logStdTrainable=False, actionClip=clip)
    policyTarget = PolicyNetworkContinuous(sess, inputLength, outputLength, args.hidden_layers_policy, nextObsPh, aPh, "Target", logStdInit=logStdInit, logStdTrainable=False, actionClip=clip)
    Q = QNetwork(sess, inputLength, outputLength, args.hidden_layers_q, obsPh, aPh, suffix="Orig") # original Q network
    QAux = QNetwork(sess, inputLength, outputLength, args.hidden_layers_q, obsPh, policy.actionFinal, suffix="Aux", reuse=Q) # network with parameters same as original Q network, but instead of action placeholder it takse output from current policy
    QTarget = QNetwork(sess, inputLength, outputLength, args.hidden_layers_q, nextObsPh, policyTarget.actionFinal,suffix="Target") #target Q network, instead of action placeholder it takse output from target policy

    #definition of losses to optimize
    policyLoss = -tf.reduce_mean(QAux.output)# - sign because we want to maximize our objective    
    targets = tf.stop_gradient(rewPh + args.gamma*(1-terPh)*QTarget.output)#check dimensions
    qLoss = tf.reduce_mean((Q.output - targets)**2)
    
    qOptimizationStep = tf.train.AdamOptimizer(learning_rate = args.learning_rate_q).minimize(qLoss)
    policyParams = utils.get_vars("PolicyNetworkContinuousOrig")
    policyOptimizationStep = tf.train.AdamOptimizer(learning_rate = args.learning_rate_policy).minimize(policyLoss, var_list=policyParams)

    #tf session initialization
    init = tf.initialize_local_variables()
    init2 = tf.initialize_all_variables()
    sess.run([init,init2])
    
    #algorithm
    finishedEp = 0
    step = 0
    buffer = ReplayBuffer(args.buffer_size)
    while step < args.total_train_steps:  

        obs, epLen, epRet, doSample = env.reset(), 0, 0, True

        #basicaly this is one episode because while exits when terminal state is reached or max number of steps(in episode or generaly) is reached
        while doSample:              
            print("Step {}".format(step))
            sampledAction, _, _, _ = policy.getSampledActions(np.expand_dims(obs, 0))      
            nextObs, reward, terminal, _ = env.step(sampledAction[0])  
            epLen += 1
            epRet += reward

            buffer.add(np.expand_dims(obs, 0), sampledAction, reward, np.expand_dims(nextObs, 0), terminal)
            obs = nextObs

            doSample = not terminal and epLen < args.max_episode_len and step < args.total_train_steps
            
            if terminal and args.exp_name is not None:
                summaryRet, summaryLen = sess.run([epRewSum, epLenSum], feed_dict = {epRewPh:epRet, epLenPh:epLen})
                writer.add_summary(summaryRet, finishedEp)
                writer.add_summary(summaryLen, finishedEp)                    
                finishedEp += 1
            
            step +=1          
            
            #time for update
            if step > args.update_after and step % args.update_freq == 0:           
                observations, actions, rewards, nextObservations, terminals = buffer.sample(args.batch_size)                

                sess.run(qOptimizationStep, feed_dict={obsPh:observations, nextObsPh:nextObservations, rewPh:rewards, terPh:terminals, aPh:actions})
                sess.run(policyOptimizationStep, feed_dict={obsPh:observations})

                utils.polyak(QTarget, Q, args.rho, sess, verbose=True)
                utils.polyak(policyTarget, policy, args.rho, sess, verbose=True)
    
    writer.close()
    env.close()
    