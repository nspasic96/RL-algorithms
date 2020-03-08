import argparse
import gym
import numpy as np
import tensorflow as tf
import time
import wandb

import utils
from networks import StateValueNetwork, PolicyNetworkDiscrete, PolicyNetworkContinuous
from GAEBuffer import GAEBuffer

parser = argparse.ArgumentParser(description='TRPO')

parser.add_argument('--gym-id', type=str, default="Pong-ram-v0",
                   help='the id of the gym environment')
parser.add_argument('--seed', type=int, default=0,
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
                   help='the learning rate of the optimizer of state-value function')
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
args = parser.parse_args()

if args.exp_name is not None:
    wandb.init(project=args.wandb_projet_name, config=vars(args), name=args.exp_name)

if not args.seed:
    args.seed = int(time.time())
    
sess = tf.Session()
env = gym.make(args.gym_id)
discreteActionsSpace = utils.isDiscrete(env)

inputLength = env.observation_space.shape[0]
outputLength = env.action_space.n if discreteActionsSpace else env.action_space.shape[0]


#definition of placeholders
logProbSampPh = tf.placeholder(dtype = tf.float32, shape=[None], name="logProbSampled") #log probabiliy of action sampled from sampling distribution (pi_old)
advPh = tf.placeholder(dtype = tf.float32, shape=[None], name="advantages") #advantages obtained using GAE-lambda and values obtainet from StateValueNetwork V
returnsPh = tf.placeholder(dtype = tf.float32, shape=[None], name="returns") #total discounted cumulative reward
policyParamsFlatten= tf.placeholder(dtype = tf.float32, shape=[None], name = "policyParams") #policy params flatten, used in assingment of pi params in line search algorithm
obsPh = tf.placeholder(dtype=tf.float32, shape=[None, inputLength], name="observations") #observations

if discreteActionsSpace:
    aPh = tf.placeholder(dtype=tf.int32, shape=[None], name="actions") #actions taken
    logProbsAllPh = tf.placeholder(dtype= tf.float32, shape=[None, outputLength], name="logProbsAll") #log probabilities of all actions according to sampling distribution (pi_old)
    additionalInfoLengths = [outputLength]
else:
    aPh = tf.placeholder(dtype=tf.float32, shape=[None,outputLength], name="actions")
    oldActionMeanPh = tf.placeholder(dtype=tf.float32, shape=[None,outputLength], name="actionsMeanOld")
    oldActionLogStdPh = tf.placeholder(dtype=tf.float32, shape=[None,outputLength], name="actionsLogStdOld") 
    additionalInfoLengths = [outputLength, outputLength]

buffer = GAEBuffer(args.gamma, args.lambd, args.epoch_len, inputLength, 1 if discreteActionsSpace else outputLength, additionalInfoLengths)

#definition of networks
V = StateValueNetwork(sess, inputLength, args.learning_rate_state_value, obsPh) #this network has method for training, but it is never used. Instead, training is done outside of this class
if(discreteActionsSpace):
    policy = PolicyNetworkDiscrete(sess, inputLength, outputLength, obsPh, aPh) #policy network for discrete action space
else:      
    policy = PolicyNetworkContinuous(sess, inputLength, outputLength, obsPh, aPh)
    
#definition of losses to optimize
ratio = tf.exp(policy.logProbWithCurrParams - logProbSampPh)
Lloss = -tf.reduce_mean(ratio*advPh) # - sign because we want to maximize our objective
stateValueLoss = tf.reduce_mean((V.output - returnsPh)**2)

KLinputs={}
if(discreteActionsSpace):
    KLcontraint = utils.categorical_kl(policy.logProbs, logProbsAllPh) 
else:
    KLcontraint = utils.diagonal_gaussian_kl(policy.actionMean, policy.actionLogStd, oldActionMeanPh, oldActionLogStdPh)     

svfOptimizationStep = tf.train.AdamOptimizer(learning_rate = args.learning_rate_state_value).minimize(stateValueLoss)

#tf session initialization
init = tf.initialize_local_variables()
init2 = tf.initialize_all_variables()
sess.run([init,init2])

#algorithm
for e in range(args.epochs):
    
    obs, epLen, epRet = env.reset(), 0, 0
    
    epochSt = time.time()
    for l in range(args.epoch_len):
        
        if(e%10 == 0):
            env.render()
        
        if discreteActionsSpace:
            sampledAction, logProbSampledAction, logProbsAll = policy.getSampledActions(np.expand_dims(obs, 0))
            additionalInfos = [logProbsAll]
        else:
            sampledAction, logProbSampledAction, actionsMean, actionLogStd = policy.getSampledActions(np.expand_dims(obs, 0))
            additionalInfos = [actionsMean, actionLogStd]
            
        predictedV = V.forward(np.expand_dims(obs, 0))
        
        nextObs, reward, terminal, _ = env.step(sampledAction[0])  
        epLen += 1
        epRet += reward

        buffer.add(obs, sampledAction[0], predictedV[0][0], logProbSampledAction, reward, additionalInfos)
        obs = nextObs

        done = terminal or l == args.max_episode_len
        if(done or l == args.epoch_len -1):
            #if(not terminal):
                #print("Cutting path. Either max episode length steps are done in current episode or epoch has finished")
            val = 0 if terminal else V.forward(np.expand_dims(obs, 0))
            buffer.finishPath(val)
            if terminal and args.exp_name is not None:
                wandb.log({'Total reward': epRet, 'Episode length':epLen})
            obs, epLen, epRet = env.reset(), 0, 0
    
    epochEnd = time.time()
    print("Epoch {} ended in {}".format(e, epochEnd-epochSt))
        
    #update policy and update state-value(multiple times) after that
    observations, actions, advEst, sampledLogProb, returns, additionalInfos = buffer.get()
    
    suffix = "Continuous"
    if discreteActionsSpace:
        suffix = "Discrete"
    policyParams = utils.get_vars("PolicyNetwork"+ suffix)
    getPolicyParams = utils.flat_concat(policyParams)
    setPolicyParams = utils.assign_params_from_flat(policyParamsFlatten, policyParams)
    
    d, HxOp = utils.hesian_vector_product(KLcontraint, policyParams)
    
    if args.damping_coef > 0:
        HxOp += args.damping_coef * d
    if discreteActionsSpace:
        Hx = lambda newDir : sess.run(HxOp, feed_dict={d : newDir, logProbsAllPh : additionalInfos[0], obsPh : observations})
    else:
        Hx = lambda newDir : sess.run(HxOp, feed_dict={d : newDir, oldActionMeanPh : additionalInfos[0], oldActionLogStdPh : additionalInfos[1], obsPh : observations})
    
    grad = sess.run(utils.flat_grad(Lloss, policyParams), feed_dict = { obsPh : observations, aPh: actions, advPh : advEst, logProbSampPh : sampledLogProb})#, logProbsAllPh : allLogProbs})
    newDir = utils.conjugate_gradients(Hx, grad, args.cg_iters)

    LlossOld = sess.run(Lloss , feed_dict={obsPh : observations, aPh: actions, advPh : advEst, logProbSampPh : sampledLogProb})#, logProbsAllPh : allLogProbs})#L function inputs: observations, advantages estimated, logProb of sampled action, logProbsOfAllActions              
    
    coef = np.sqrt(2*args.delta/(np.dot(np.transpose(newDir),Hx(newDir)) + 1e-8))
    
    oldParams = sess.run(getPolicyParams)
    
    kl = 0
    
    for j in range(args.max_iters_line_search):
        step = args.alpha**j
        
        #check if KL distance is within limits and is Lloss going down
        sess.run(setPolicyParams, feed_dict={policyParamsFlatten : oldParams - coef*newDir*step})
        if discreteActionsSpace:
            kl, LlossNew =  sess.run([KLcontraint, Lloss],  feed_dict={obsPh : observations, aPh: actions, advPh : advEst, logProbSampPh : sampledLogProb, logProbsAllPh : additionalInfos[0]})#same as L function inputs plus
        else:
            kl, LlossNew =  sess.run([KLcontraint, Lloss],  feed_dict={obsPh : observations, aPh: actions, advPh : advEst, logProbSampPh : sampledLogProb, oldActionMeanPh : additionalInfos[0], oldActionLogStdPh : additionalInfos[1]})#same as L function inputs plus
        
        if (kl <= args.delta and LlossNew <= LlossOld):
            LlossOld = LlossNew
            oldParams = oldParams - coef*newDir*step
            break
        
        if(j == args.max_iters_line_search - 1):
            sess.run(setPolicyParams, feed_dict={policyParamsFlatten : oldParams})
            print("Line search didn't find step size that satisfies KL constraint")
    
    
    for j in range(args.state_value_network_updates):
        sess.run(svfOptimizationStep, feed_dict={obsPh : observations, returnsPh : returns})
    
    SVLoss = sess.run(stateValueLoss, feed_dict={obsPh : observations, returnsPh : returns})
    if args.exp_name is not None:
        wandb.log({'Value function loss': SVLoss, 
                   'Surrogate function value': LlossOld, 
                   'KL divergence': kl})
    