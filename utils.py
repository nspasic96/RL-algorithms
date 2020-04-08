import numpy as np
import tensorflow as tf
from scipy.stats import norm as NormalDistribution #TODO:check if this should be here
import scipy.signal as signal
from gym.spaces import Box, Discrete

def get_batch_from_memory(idxs, memory, target_network, q_network, GAMMA):

    states=[]
    states_next=[]
    targets =[]
    actions = []
    rewards = []
    dones = []

    for idx in idxs:

        state, action, reward, state_next, done = memory[idx]
        states.append(np.expand_dims(state,0))
        states_next.append(np.expand_dims(state_next,0))
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        #if(done):
            #print(reward)

    states = np.concatenate(states, axis=0)
    #print("Input to nn is of shape {}".format(states.shape))
    state_next = np.concatenate(states_next, axis=0)

    outputs = target_network.predict(state_next)
    target_f = q_network.predict(states)

    for i in range(target_f.shape[0]):
        target = rewards[i] + (1-dones[i])*GAMMA*np.amax(outputs[i])
        #print("Target is {}".format(target))
        target_f[i][actions[i]] = target

    targets = target_f

    return states, targets

def addToReplyBuffer(buffer, instance, idx, max_size):
    if(idx < max_size):
        buffer.append(instance)
    else:
        idx = idx % max_size
        buffer[idx] = instance
        
def prepareInputs(sampledBatch, gamma, squash):
        
    observations = sampledBatch[0][0]
    if squash:
        actionsTaken = sampledBatch[0][1][0]
    else:
        actionsTaken = sampledBatch[0][1][1]
    rewards = sampledBatch[0][2]
    stateValuesAux = sampledBatch[0][3]
    minQs = sampledBatch[0][4]
        
    actionRaw = sampledBatch[0][1][1]
    actionMeans = sampledBatch[0][1][2]
    actionStds = sampledBatch[0][1][3]
    
    logPi1 = np.log(NormalDistribution(actionMeans[0, 0],actionStds[0, 0]).pdf(actionRaw[0, 0]))
    logPi2 = np.log(NormalDistribution(actionMeans[0, 1],actionStds[0, 1]).pdf(actionRaw[0, 1]))
    logPi3 = np.log(NormalDistribution(actionMeans[0, 2],actionStds[0, 2]).pdf(actionRaw[0, 2]))
    logPis = [logPi1 + logPi2 + logPi3]
    
    for i in range(1,len(sampledBatch)):
        
        observations = np.vstack((observations, sampledBatch[i][0]))
        if squash:
            actionsTaken = np.vstack((actionsTaken, sampledBatch[i][1][0]))
        else:
            actionsTaken = np.vstack((actionsTaken, sampledBatch[i][1][1]))
            
        rewards = np.vstack((rewards, sampledBatch[i][2]))
        stateValuesAux = np.vstack((stateValuesAux, sampledBatch[i][3]))
        minQs = np.vstack((minQs, sampledBatch[i][4]))
            
        actionRaw = sampledBatch[i][1][1]
        actionMeans = sampledBatch[i][1][2]
        actionStds = sampledBatch[i][1][3]      
        
        logPi1 = np.log(NormalDistribution(actionMeans[0, 0],actionStds[0, 0]).pdf(actionRaw[0, 0]))
        logPi2 = np.log(NormalDistribution(actionMeans[0, 1],actionStds[0, 1]).pdf(actionRaw[0, 1]))
        logPi3 = np.log(NormalDistribution(actionMeans[0, 2],actionStds[0, 2]).pdf(actionRaw[0, 2]))
        curr = [logPi1 + logPi2 + logPi3]
        
        logPis = np.vstack((logPis, curr))
                
    QsHat = rewards + gamma*stateValuesAux
        
    return observations, logPis, actionsTaken, rewards, minQs, QsHat

def conjugate_gradients(Ax, b, cg_iters):
    """
    Conjugate gradient algorithm
    (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
    """
    x = np.zeros_like(b)
    r = b.copy() # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
    p = r.copy()
    r_dot_old = np.dot(r,r)
    for _ in range(cg_iters):
        z = Ax(p)
        alpha = r_dot_old / (np.dot(p, z) + 1e-8)
        x += alpha * p
        r -= alpha * z
        r_dot_new = np.dot(r,r)
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new
    return x

def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]

def flat_concat(xs):
    return tf.concat([tf.reshape(x,(-1,)) for x in xs], axis=0)

def flat_grad(f, params):
    return flat_concat(tf.gradients(xs=params, ys=f))

def assign_params_from_flat(x, params):
    flat_size = lambda p : int(np.prod(p.shape.as_list())) # the 'int' is important for scalars
    splits = tf.split(x, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    return tf.group([tf.assign(p, p_new) for p, p_new in zip(params, new_params)])

def hesian_vector_product(f, theta):
    g = flat_grad(f, theta)
    x = tf.placeholder(dtype=tf.float32, shape=g.shape, name="Hinvg")
    gTx = tf.reduce_sum(g*x)
    Hx = flat_grad(gTx, theta)
    return x, Hx     

def disount_cumsum(x, discount):
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def diagonal_gaussian_kl(mu0, log_std0, mu1, log_std1):
    """
    tf symbol for mean KL divergence between two batches of diagonal gaussian distributions,
    where distributions are specified by means and log stds.
    (https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Multivariate_normal_distributions)
    """
    var0, var1 = tf.exp(2 * log_std0), tf.exp(2 * log_std1)
    pre_sum = 0.5*(((mu1- mu0)**2 + var0)/(var1 + 1e-8) - 1) +  log_std1 - log_std0
    all_kls = tf.reduce_sum(pre_sum, axis=1)
    return tf.reduce_mean(all_kls)

def categorical_kl(logp0, logp1):
    """
    tf symbol for mean KL divergence between two batches of categorical probability distributions,
    where the distributions are input as log probs.
    """
    all_kls = tf.reduce_sum(tf.exp(logp1) * (logp1 - logp0), axis=1)
    return tf.reduce_mean(all_kls)

def is_discrete(env):
    return isinstance(env.action_space, Discrete)

def polyak(dst, src, rho, session, verbose):
    scopeName= src.variablesScope
    trainableVars = tf.trainable_variables(scopeName)
    assignOps=[]

    if(verbose):
        print("Scope name source: {}\nScope name destination: {}\nVariables to apply polyak:".format(scopeName,dst.variablesScope))
        
    for var in trainableVars:
        otherNamePart = var.name.split(scopeName)[1]
        dstVar = session.graph.get_tensor_by_name(dst.variablesScope+otherNamePart)
        if(verbose):
            print(dstVar)
        newVal = rho*var+(1-rho)*dstVar
        assignOps.append(tf.assign(dstVar, newVal))
    
    return assignOps
    