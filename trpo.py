import argparse
import gym
import numpy as np
import tensorflow as tf
import time

import utils
from networks import StateValueNetwork, PolicyNetworkDiscrete
from GAEBuffer import GAEBuffer


parser = argparse.ArgumentParser(description='TRPO')

parser.add_argument('--gym-id', type=str, default="Pong-ram-v0",
                   help='the id of the gym environment')
parser.add_argument('--learning-rate-state-value', type=float, default=7e-4,
                   help='the learning rate of the optimizer of state-value function')
parser.add_argument('--seed', type=int, default=1,
                   help='seed of the experiment')
parser.add_argument('--prod-mode', type=bool, default=False,
                   help='run the script in production mode and use wandb to log outputs')
parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                   help="the wandb's project name")
parser.add_argument('--buffer-size', type=int, default=50000,
                    help='memory buffer size')
parser.add_argument('--gamma', type=float, default=0.99,
                   help='the discount factor gamma')
parser.add_argument('--lambda', type=float, default=0.97,
                   help='USED FOR GAE LAMBDA')
parser.add_argument('--delta', type=float, default=0.05,
                   help='max KL distance between two successive distributions')
parser.add_argument('--state-value-network-updates', type=int, default=10,
                   help="number of updates for state-value network")
parser.add_argument('--epochs', type=int, default=50,
                   help="epochs to train")
parser.add_argument('--epoch_len', type=int, default=2000,
                   help="length of one epoch")
parser.add_argument('--max_episode_len', type=int, default=1000,
                   help="max length of one episode")
parser.add_argument('--alpha', type=float, default=0.8,
                   help="defult step size in line serach")
parser.add_argument('--max_iters_line_search', type=int, default=10,
                   help="maximum steps to take in line serach")
args = parser.parse_args()

if not args.seed:
    args.seed = int(time.time())
    
sess = tf.Session()
env = gym.make(args.gym_id)
buffer = GAEBuffer()

inputShape = env.observation_space.shape[0]
outputShape = env.action_space.shape[0]

#definition of placeholders
logSampDistPh = tf.placeholder(dtype = tf.float32, shape=[None]) #log probabiliy of action from sampling distribution (pi_old)
advPh = tf.placeholder(dtype = tf.float32, shape=[None])
vPh = tf.placeholder(dtype = tf.float32, shape=[None])
policyParamsFlatten= tf.placeholder(dtype = tf.float32, shape=[None])

#definition of networks
V = StateValueNetwork(sess, inputShape, args.learningRateStateValue)
policy = PolicyNetworkDiscrete(sess, inputShape, outputShape)

#definition of losses to optimize
ratio = tf.exp(policy.logProbs - logSampDistPh)
Lloss = -tf.reduce_mean(ratio*advPh) # - sign because we want to maximize our objective
stateValueLoss = tf.reduce_mean((V.output - vPh)**2)
KLcontraint = tf.reduce_mean(tf.reduce_sum((logSampDistPh - policy.logProbs)*tf.exp(logSampDistPh), axis=1))

svfOptimizationStep = tf.train.AdamOptimizer(learning_rate = args.learningRateStateValue).minimize(stateValueLoss)

#algorithm
for e in range(args.epochs):
    
    obs = env.reset()
    
    for l in range(args.epochLen):
        
        sampledAction, logProbSampledAction = policy.getSampledActions(np.expand_dims(obs, 0))    
        predictedValue = V.forward(np.expand_dims(obs, 0))
        
        nextObs, reward, terminal, _ = env.step(sampledAction[0])  
        
        buffer.append(obs, predictedValue, reward, sampledAction, logProbSampledAction)
        obs = nextObs

        done = terminal or l == args.maxEpisodeLen
        
        if(done or l == args.epochLen -1):
            if(not terminal):
                print("Cutting path")
            val = 0 if terminal else V.forward(np.expand_dims(obs, 0))
            buffer.finishPath(val)
            obs = env.reset()       
        
    #update policy and update state-value after that
    
    _,_,_,..._ = buffer.get()
    
    policyParams = utils.get_vars("PolicyNetworkDiscrete")
    getPolicyParams = utils.flat_concat(policyParams)
    setPolicyParams = utils.assign_params_from_flat(policyParamsFlatten, policyParams)
    
    x, HxOp = utils.hesian_vector_product(KLcontraint, policyParams)
    Hx = lambda newDir : sess.run(HxOp, feed_dict={x : newDir})
    
    grad = sess.run(utils.flat_grad(KLcontraint, policyParams))
    newDir = utils.conjugateGradients(Hx, grad)

    LlossOld = sess.run(Lloss,feed_dict={})             
    coef = np.sqrt(2*args.delta/np.dot(np.transpose(newDir),Hx(newDir)))
    
    oldParams = sess.run(getPolicyParams, feed_dict={})
    
    for j in range(args.maxItersLineSearch):
        step = args.alpha**j
        
        #check if KL distance is within limits and is L_loss going down
        sess.run(setPolicyParams, feed_dict={policyParamsFlatten : oldParams - coef*newDir*step})
        kl, LlossNew =  sess.run([KLcontraint, Lloss], feed_dict={})
    
        if (kl <= args.delta and LlossNew <= LlossOld):
            L_loss_old = LlossNew
            old_params = oldParams - coef*newDir*step
            break
        
        if(j == args.maxItersLineSearch - 1):
            sess.run(setPolicyParams, feed_dict={policyParamsFlatten : oldParams})
            print("Line search didn't find step size that satisfies KL constraint")
    
    for j in range(args.stateValueNetworkUpdates):
        sess.run(svfOptimizationStep, feed_dict={})
            
    
    
        




































