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
parser.add_argument('--seed', type=int, default=0,
                   help='seed of the experiment')
parser.add_argument('--epochs', type=int, default=50,
                   help="epochs to train")
parser.add_argument('--epoch_len', type=int, default=4000,
                   help="length of one epoch")
parser.add_argument('--gamma', type=float, default=0.99,
                   help='the discount factor gamma')
parser.add_argument('--delta', type=float, default=0.01,
                   help='max KL distance between two successive distributions')
parser.add_argument('--learning_rate_state_value', type=float, default=1e-3,
                   help='the learning rate of the optimizer of state-value function')
parser.add_argument('--state-value-network-updates', type=int, default=80,
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
args = parser.parse_args()

if not args.seed:
    args.seed = int(time.time())
    
sess = tf.Session()
env = gym.make(args.gym_id)

inputLength = env.observation_space.shape[0]
outputLength = env.action_space.n

buffer = GAEBuffer(args.gamma, args.lambd, args.epoch_len, inputLength, outputLength)

#definition of placeholders
logProbSampPh = tf.placeholder(dtype = tf.float32, shape=[None], name="logProbSampled") #log probabiliy of action sampled from sampling distribution (pi_old)
logProbsAllPh = tf.placeholder(dtype= tf.float32, shape=[None, outputLength], name="logProbsAll") #log probabilities of all actions according to sampling distribution (pi_old)
advPh = tf.placeholder(dtype = tf.float32, shape=[None], name="advantages") #advantages obtained using GAE-lambda and values obtainet from StateValueNetwork V
returnsPh = tf.placeholder(dtype = tf.float32, shape=[None], name="returns") #total discounted cumulative reward
policyParamsFlatten= tf.placeholder(dtype = tf.float32, shape=[None], name = "policyParams") #policy params flatten, used in assingment of pi params in line search algorithm
obsPh = tf.placeholder(dtype=tf.float32, shape=[None, inputLength], name="observations") #observations
aPh = tf.placeholder(dtype=tf.int32, shape=[None], name="actions") #actions taken

#definition of networks
V = StateValueNetwork(sess, inputLength, args.learning_rate_state_value, obsPh) #this network has method for training, but it is never used. Instead, training is done outside of this class
policy = PolicyNetworkDiscrete(sess, inputLength, outputLength, obsPh) #policy network for discrete action space
      
#definition of losses to optimize
ratio = tf.exp(policy.logProb(aPh) - logProbSampPh)
Lloss = -tf.reduce_mean(ratio*advPh) # - sign because we want to maximize our objective
stateValueLoss = tf.reduce_mean((V.output - returnsPh)**2)
KLcontraint = tf.reduce_mean(tf.reduce_sum((logProbsAllPh - policy.logProbs)*tf.exp(logProbsAllPh), axis=1))

svfOptimizationStep = tf.train.AdamOptimizer(learning_rate = args.learning_rate_state_value).minimize(stateValueLoss)

#tf session initialization
init = tf.initialize_local_variables()
init2 = tf.initialize_all_variables()
sess.run([init,init2])

#algorithm
for e in range(args.epochs):
    
    obs = env.reset()
    
    for l in range(args.epoch_len):
        
        env.render()

        sampledAction, logProbSampledAction, logProbsAll = policy.getSampledActions(np.expand_dims(obs, 0))    
        predictedV = V.forward(np.expand_dims(obs, 0))
        
        nextObs, reward, terminal, _ = env.step(sampledAction[0])  

        buffer.add(obs, sampledAction[0], predictedV, logProbSampledAction, logProbsAll, reward)
        obs = nextObs

        done = terminal or l == args.max_episode_len
        if(done or l == args.epoch_len -1):
            if(not terminal):
                print("Cutting path. Either max episode length steps are done in current episode or epoch has finished")
            val = 0 if terminal else V.forward(np.expand_dims(obs, 0))
            buffer.finishPath(val)
            obs = env.reset()      
        
    #update policy and update state-value(multiple times) after that
    observations, actions, advEst, sampledLogProb, allLogProbs, returns = buffer.get()
    
    policyParams = utils.get_vars("PolicyNetworkDiscrete")
    getPolicyParams = utils.flat_concat(policyParams)
    setPolicyParams = utils.assign_params_from_flat(policyParamsFlatten, policyParams)
    
    d, HxOp = utils.hesian_vector_product(KLcontraint, policyParams)#this is strange, newton direction with respect to constraint?
    if args.damping_coef > 0:
        HxOp += args.damping_coef * d
    Hx = lambda newDir : sess.run(HxOp, feed_dict={d : newDir, logProbsAllPh : allLogProbs, obsPh : observations})
    
    grad = sess.run(utils.flat_grad(KLcontraint, policyParams), feed_dict = { logProbsAllPh : allLogProbs, obsPh:observations})
    newDir = utils.conjugate_gradients(Hx, grad, args.cg_iters)

    LlossOld = sess.run(Lloss , feed_dict={obsPh : observations, aPh: actions, advPh : advEst, logProbSampPh : sampledLogProb, logProbsAllPh : allLogProbs})#L function inputs: observations, advantages estimated, logProb of sampled action, logProbsOfAllActions              
    coef = np.sqrt(2*args.delta/(np.dot(np.transpose(newDir),Hx(newDir)) + 1e-8))
    
    oldParams = sess.run(getPolicyParams)
    
    for j in range(args.max_iters_line_search):
        step = args.alpha**j
        
        #check if KL distance is within limits and is Lloss going down
        sess.run(setPolicyParams, feed_dict={policyParamsFlatten : oldParams - coef*newDir*step})
        kl, LlossNew =  sess.run([KLcontraint, Lloss],  feed_dict={obsPh : observations, aPh: actions, advPh : advEst, logProbSampPh : sampledLogProb, logProbsAllPh : allLogProbs})#same as L function inputs plus
    
        if (kl <= args.delta and LlossNew <= LlossOld):
            L_loss_old = LlossNew
            old_params = oldParams - coef*newDir*step
            break
        
        if(j == args.max_iters_line_search - 1):
            sess.run(setPolicyParams, feed_dict={policyParamsFlatten : oldParams})
            print("Line search didn't find step size that satisfies KL constraint")
    
    for j in range(args.state_value_network_updates):
        sess.run(svfOptimizationStep, feed_dict={obsPh:observations, returnsPh : returns})