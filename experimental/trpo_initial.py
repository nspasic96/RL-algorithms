import argparse
import gym
import pybulletgym
import numpy as np
import tensorflow as tf
import time
import wandb

import sys
sys.path.append("../")

import utils
from networks import StateValueNetwork, PolicyNetworkDiscrete, PolicyNetworkContinuous
from GAEBuffer import GAEBuffer

parser = argparse.ArgumentParser(description='TRPO')

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
parser.add_argument('--wandb_log', type=bool, default=False,
                   help='whether to log results to wandb')
parser.add_argument('--play_every_nth_epoch', type=int, default=10,
                   help='every nth epoch will be rendered, set to epochs+1 not to render at all')
args = parser.parse_args()

svfTrainInBatchesThreshold = 5000

if not args.seed:
    args.seed = int(time.time())
    
graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    env = gym.make(args.gym_id)
    discreteActionsSpace = utils.is_discrete(env)
    
    inputLength = env.observation_space.shape[0]
    outputLength = env.action_space.n if discreteActionsSpace else env.action_space.shape[0]
    
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
            
    if args.wandb_log:
        
        implSuffix = "initital"
        experimentName = f"{args.gym_id}__trpo_{implSuffix}__{args.seed}__{int(time.time())}"
        writer = tf.summary.FileWriter(f"runs/{experimentName}", graph = sess.graph)
        cnf = vars(args)
        cnf['action_space_type'] = 'discrete' if discreteActionsSpace else 'continuous'
        cnf['input_length'] = inputLength
        cnf['output_length'] = outputLength
        cnf['exp_name_tb'] = experimentName
        note = ""
        if(args.epoch_len > svfTrainInBatchesThreshold):
            note = "State value training split in batches of size {} because of too large number of samples".format(svfTrainInBatchesThreshold)
        wandb.init(project=args.wandb_projet_name, config=cnf, name=experimentName, notes=note, tensorboard=True)
    
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
    V = StateValueNetwork(sess, inputLength, args.hidden_layers_state_value, args.learning_rate_state_value, obsPh) #this network has method for training, but it is never used. Instead, training is done outside of this class
    if(discreteActionsSpace):
        policy = PolicyNetworkDiscrete(sess, inputLength, outputLength, args.hidden_layers_policy, obsPh, aPh, "Orig") #policy network for discrete action space
    else:          
        policyActivations = [tf.nn.relu for i in range(len(args.hidden_layers_policy))] + [None]    
        policy = PolicyNetworkContinuous(sess, inputLength, outputLength, args.hidden_layers_policy, policyActivations, obsPh, aPh, "Orig", logStdInit=-0.5*np.ones((1,outputLength), dtype=np.float32))
      
    #definition of losses to optimize
    ratio = tf.exp(policy.logProbWithCurrParams - logProbSampPh)
    Lloss = -tf.reduce_mean(ratio*advPh) # - sign because we want to maximize our objective
    stateValueLoss = tf.reduce_mean((V.output - returnsPh)**2)
    
    if(discreteActionsSpace):
        KLcontraint = utils.categorical_kl(policy.logProbs, logProbsAllPh) 
    else:
        KLcontraint = utils.diagonal_gaussian_kl(policy.actionMean, policy.actionLogStd, oldActionMeanPh, oldActionLogStdPh)     
    
    svfOptimizationStep = tf.train.AdamOptimizer(learning_rate = args.learning_rate_state_value).minimize(stateValueLoss)
    
    #other ops
    suffix = "Continuous"
    if discreteActionsSpace:
        suffix = "Discrete"
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
    for e in range(args.epochs):    
        print("Epoch {} started".format(e))
        obs, epLen, epRet = env.reset(), 0, 0
        
        epochSt = time.time()
        for l in range(args.epoch_len):
            
            if(e % args.play_every_nth_epoch == args.play_every_nth_epoch - 1):
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
    
            done = terminal or epLen == args.max_episode_len
            if(done or l == args.epoch_len -1):
                #if(not terminal):
                    #print("Cutting path. Either max episode length steps are done in current episode or epoch has finished")
                val = 0 if terminal else V.forward(np.expand_dims(obs, 0))
                buffer.finishPath(val)
                if terminal and args.wandb_log:
                    summaryRet, summaryLen = sess.run([epRewSum, epLenSum], feed_dict = {epRewPh:epRet, epLenPh:epLen})
                    writer.add_summary(summaryRet, finishedEp)
                    writer.add_summary(summaryLen, finishedEp)                    
                    finishedEp += 1
                obs, epLen, epRet = env.reset(), 0, 0 
        simulationEnd = time.time()      
        
        print("\tSimulation in epoch {} finished in {}".format(e, simulationEnd-epochSt))  
            
        #update policy and update state-value(multiple times) after that
        observations, actions, advEst, sampledLogProb, returns, additionalInfos = buffer.get()
            
        policyUpdateStart = time.time()  
        if discreteActionsSpace:
            Hx = lambda newDir : sess.run(HxOp, feed_dict={d : newDir, logProbsAllPh : additionalInfos[0], obsPh : observations})
        else:
            Hx = lambda newDir : sess.run(HxOp, feed_dict={d : newDir, oldActionMeanPh : additionalInfos[0], oldActionLogStdPh : additionalInfos[1], obsPh : observations})
            
        grad = sess.run(utils.flat_grad(Lloss, policyParams), feed_dict = { obsPh : observations, aPh: actions, advPh : advEst, logProbSampPh : sampledLogProb})#, logProbsAllPh : allLogProbs})
        cjStart = time.time()
        newDir = utils.conjugate_gradients(Hx, grad, args.cg_iters)    
        cjEnd = time.time()
        print("\t\tConjugate gradient in epoch {} done in {}".format(e, cjEnd-cjStart)) 
        
        LlossOld = sess.run(Lloss , feed_dict={obsPh : observations, aPh: actions, advPh : advEst, logProbSampPh : sampledLogProb})#, logProbsAllPh : allLogProbs})#L function inputs: observations, advantages estimated, logProb of sampled action, logProbsOfAllActions              
        
        coef = np.sqrt(2*args.delta/(np.dot(np.transpose(newDir),Hx(newDir)) + 1e-8))
        
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
            
            if (kl <= args.delta and LlossNew <= LlossOld):
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
            #5000 works as batch size, 10000 doesn't. For now, training is split so that no input exceeds 5000 examples
            perm = np.random.permutation(total)
            if(total > svfTrainInBatchesThreshold):
                start = 0
                while(start < total):    
                    end = np.amin([start+svfTrainInBatchesThreshold, total])
                    sess.run(svfOptimizationStep, feed_dict={obsPh : observations[perm[start:end]], returnsPh : returns[perm[start:end]]})
                    start = end
            else:
                sess.run(svfOptimizationStep, feed_dict={obsPh : observations[perm], returnsPh : returns[perm]})
         
        svfUpdateEnd = time.time()     
        
        print("\tState value in epoch {} updated in {}".format(e, svfUpdateEnd-svfUpdateStart))    
           
        #after update, calculate new loss
        svLossStart = time.time()
        if observations.shape[0] > 2*svfTrainInBatchesThreshold:
            start = 0
            SVLossSum = 0
            while(start < total):    
                end = np.amin([start+2*svfTrainInBatchesThreshold, total])
                avgBatch = sess.run(stateValueLoss, feed_dict={obsPh : observations[start:end], returnsPh : returns[start:end]})
                SVLossSum += avgBatch*(end-start)
                start = end
            SVLoss = SVLossSum/observations.shape[0]
        else:
            SVLoss = sess.run(stateValueLoss, feed_dict={obsPh : observations, returnsPh : returns})            
           
        svLossEnd = time.time()      
        
        print("\tState value loss in epoch {} calculated in {}".format(e, svLossEnd-svLossStart))
        
        if args.wandb_log:       
            summarySVm, summarySurrogateDiff, summaryKL = sess.run([SVLossSummary, SurrogateDiffSum, KLSum], feed_dict = {SVLossPh:SVLoss, SurrogateDiffPh:LlossNew - LlossOld, KLPh:kl})
            writer.add_summary(summarySVm, e)
            writer.add_summary(summarySurrogateDiff, e)
            writer.add_summary(summaryKL, e)        
        
        epochEnd = time.time()
        print("Epoch {} ended in {}".format(e, epochEnd-epochSt))
    
    writer.close()
    env.close()
    