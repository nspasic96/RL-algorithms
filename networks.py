import tensorflow as tf
import numpy as np
import utils

#TODO : For every network change train function interface: it should get placeholder names and values as input. Also change initialization(input placeholder must be provaded as well as hidden layer sizes)

tfDtype = tf.float32 
class StateValueNetwork:
    
    def __init__(self, sess, inputLength, hiddenLaySizes, learningRate, inputPh, suffix=""):
        self.inputLength = inputLength
        self.hiddenLayers = hiddenLaySizes
        self.learningRate = learningRate
        self.sess = sess
        self.global_step = tf.Variable(0,dtype = tf.int32)
        self.i = 0
        self.suffix = suffix
        self.input = inputPh
        self._createDefault()

    def _createDefault(self):
        with tf.variable_scope("StateValueNetwork{}".format(self.suffix)):
            
            curNode = tf.layers.Dense(self.hiddenLayers[0], tf.nn.tanh, use_bias = True,  name="fc1")(self.input)
            for i,l in enumerate(self.hiddenLayers[1:]):
                curNode = tf.layers.Dense(l, tf.nn.tanh, use_bias = True,  name="fc{}".format(i+2))(curNode)
            
            self.output = tf.layers.Dense(1, use_bias = False, name="output")(curNode)
            
            self.target = tf.placeholder(dtype = tfDtype, shape = [None, 1], name="target")
            self.loss = tf.losses.mean_squared_error(self.target, self.output)
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learningRate)
            
            #self.minimizationOperation = self.optimizer.minimize(self.loss, global_step = self.global_step)
            #self.gradientNorm = self.optimizer.compute_gradients(self.loss)            
                        
            gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, .5)
            self.minimizationOperation = self.optimizer.apply_gradients(zip(gradients, variables))
                    
    def forward(self, observations):        
        assert (len(observations.shape) == 2 and observations.shape[1] == self.inputLength)
        
        return self.sess.run(self.output, feed_dict = {self.input : observations})
    
    def train(self, observations, Qs, logPis): 
            
        targets = Qs - logPis  
        self.global_step = self.global_step + 1

        self.sess.run(self.minimizationOperation, feed_dict = {self.target : targets, self.input : observations})

    def update(self, other, tau):
        #TODO: assign moving exponential weights to this network
        return 0
    
    def weightsAssign(self, other):
        return 0

class QNetwork:
    
    def __init__(self, sess, inputLength, outputLength, hiddenLaySizes, inputPh, actionPh, suffix, reuse=None):
        self.inputLength = inputLength
        self.outputLength = outputLength
        self.hiddenLayers = hiddenLaySizes
        self.sess = sess
        self.global_step = tf.Variable(0,dtype = tf.int32)
        self.i = 0
        self.suffix = suffix
        self.input = inputPh
        self.action = actionPh
        self.reuse = reuse
        self._createDefault()

    def _createDefault(self):
        if self.reuse is not None:
            with tf.variable_scope("QNetwork{}".format(self.reuse.suffix)):
                curNode = tf.layers.Dense(self.hiddenLayers[0], tf.nn.tanh, use_bias = True, reuse=True, name="fc1")(self.input)
                for i,l in enumerate(self.hiddenLayers[1:]):
                    curNode = tf.layers.Dense(l, tf.nn.tanh, use_bias = True, reuse=True, name="fc{}".format(i+2))(curNode)
                
                self.output = tf.layers.Dense(1, use_bias = False, reuse=True, name="output")(curNode)
                self.variablesScope = "QNetwork{}".format(self.reuse.suffix) 
        else:   
            with tf.variable_scope("QNetwork{}".format(self.suffix)): 
                curNode = tf.layers.Dense(self.hiddenLayers[0], tf.nn.tanh, use_bias = True,  name="fc1")(self.input)
                for i,l in enumerate(self.hiddenLayers[1:]):
                    curNode = tf.layers.Dense(l, tf.nn.tanh, use_bias = True,  name="fc{}".format(i+2))(curNode)
                
                self.output = tf.layers.Dense(1, use_bias = False, name="output")(curNode)
                self.variablesScope = "QNetwork{}".format(self.suffix) 
                   
    def forward(self, observations):        
        assert (len(observations.shape) == 2 and observations.shape[1] == self.inputLength)
        
        return self.sess.run(self.output, feed_dict = {self.input : observations})
           
class PolicyNetworkDiscrete:
    
    def __init__(self, sess, inputLength, outputLength, hiddenLaySizes, inputsPh, actionsPh):
        self.inputLength = inputLength
        self.outputLength = outputLength
        self.hiddenLayers = hiddenLaySizes
        self.sess = sess
        self.global_step = tf.Variable(0,dtype = tf.int32)
        self.i = 0
        self.inputs = inputsPh
        self.actions = actionsPh
        self._createDefault()      
                
    def _createDefault(self):
        with tf.variable_scope("PolicyNetworkDiscrete"):
            
            curNode = tf.layers.Dense(self.hiddenLayers[0], tf.nn.tanh, use_bias = True,  name="fc1")(self.inputs)
            for i,l in enumerate(self.hiddenLayers[1:]):
                curNode = tf.layers.Dense(l, tf.nn.tanh, use_bias = True,  name="fc{}".format(i+2))(curNode)
            self.logits = tf.layers.Dense(self.outputLength, use_bias = True,  name="actions")(curNode)
            self.logProbs = tf.nn.log_softmax(self.logits)
            
            self.sampledActions = tf.squeeze(tf.random.categorical(self.logProbs, 1), axis=1)
            self.sampledLogProbs = tf.reduce_sum(tf.one_hot(self.sampledActions, depth = self.outputLength)*self.logProbs)
            
            self.logProbWithCurrParams = tf.reduce_sum(tf.one_hot(self.actions, depth=self.outputLength)*self.logProbs, axis=1)#log probs for actions given the observation(both fed with placeholder)
            
            
    def getSampledActions(self, inputs):                      
        return self.sess.run([self.sampledActions, self.sampledLogProbs, self.logProbs], feed_dict = {self.inputs : inputs})


class PolicyNetworkContinuous:
    
    def __init__(self, sess, inputLength, outputLength, hiddenLaySizes, inputPh, actionsPh, detachedLogStds=True, clipLogStd=False, squashActions=False):
        self.inputLength = inputLength
        self.outputLength = outputLength
        self.input = inputPh
        self.actions = actionsPh
        self.hiddenLayers = hiddenLaySizes
        self.sess = sess
        self.global_step = tf.Variable(0,dtype = tf.int32)
        self.i = 0
        self.clipLogStd = clipLogStd
        self.detachedLogStds = detachedLogStds
        self.squashActions = squashActions
        self._createDefault()
        
    def _createDefault(self):
        with tf.variable_scope("PolicyNetworkContinuous"):
            curNode = tf.layers.Dense(self.hiddenLayers[0], tf.nn.tanh, use_bias = True,  name="fc1")(self.input)
            for i,l in enumerate(self.hiddenLayers[1:]):
                curNode = tf.layers.Dense(l, tf.nn.tanh, use_bias = True,  name="fc{}".format(i+2))(curNode)
            self.actionMean = tf.layers.Dense(self.outputLength, use_bias = True,  name="ActionsMean")(curNode)
            if self.detachedLogStds:
                self.actionLogStd = tf.get_variable(name='ActionsLogStd', initializer=-0.5*np.ones(self.outputLength, dtype=np.float32))
            else:
                self.actionLogStd = tf.layers.Dense(self.outputLength, use_bias = True, name="ActionsLogStd")(curNode)
        
            if self.clipLogStd:
                self.actionLogStd = tf.clip_by_value(self.actionLogStd, -20, 2, name="ClipedActionsLogStd")
                
            self.actionStd = tf.math.exp(self.actionLogStd)
            
            self.actionRaw = self.actionMean + tf.random_normal(tf.shape(self.actionMean)) * self.actionStd
            
            #TODO: CHeck whether this work when squash=True(because gaussian_likelihood doesnt take it into consideration)
            if self.squashActions: 
                self.actionFinal = tf.tanh(self.actionRaw)
            else:
                self.actionFinal = self.actionRaw   
                
            self.sampledLogProbs = utils.gaussian_likelihood(self.actionRaw, self.actionMean, self.actionLogStd)
            self.logProbWithCurrParams = utils.gaussian_likelihood(self.actions, self.actionMean, self.actionLogStd)#log prob(joint, all action components are from gaussian) for action given the observation(both fed with placeholder)
            
                           
    def getSampledActions(self, observations):
        return self.sess.run([self.actionFinal, self.sampledLogProbs, self.actionMean, self.actionLogStd], feed_dict = {self.input : observations})
         
        
class SoftQNetwork:
    
    def __init__(self, sess, inputLength, outputLength, learningRate, alpha, suffix):
        self.inputLength = inputLength
        self.outputLength = outputLength
        self.learningRate = learningRate
        self.sess = sess
        self.alpha = alpha
        self.global_step = tf.Variable(0,dtype = tf.int32)
        self.i = 0
        self.initialized = False
        self.suffix = suffix
        
    def _createDefault(self):
        with tf.variable_scope("SoftQNetwork{}".format(self.suffix)):
            self.input = tf.placeholder(dtype = tfDtype, shape = [None, self.inputLength + self.outputLength], name="input")
            curNode = tf.layers.Dense(120, tf.nn.relu, use_bias = True, kernel_initializer= tf.initializers.truncated_normal(), name="fc1")(self.input)
            curNode = tf.layers.Dense(84, tf.nn.relu, use_bias = True, kernel_initializer= tf.initializers.truncated_normal(), name="fc2")(curNode)
            self.output = tf.layers.Dense(1, use_bias = True, kernel_initializer= tf.initializers.truncated_normal(), name="output")(curNode)
            
            self.target = tf.placeholder(dtype = tfDtype, shape = [None, 1], name="target")
            self.loss = tf.losses.mean_squared_error(self.target, self.output)
            
            self.minimizationOperation = tf.train.AdamOptimizer(learning_rate = self.learningRate).minimize(self.loss, global_step = self.global_step)
            
    def forward(self, observations, actions):
        if not self.initialized:
            self._createDefault()
            init = tf.initialize_local_variables()
            init2 = tf.initialize_all_variables()
            self.sess.run([init,init2])
            self.initialized = True
                
        fullInput = np.concatenate((observations, actions), axis = -1)
        
        return self.sess.run(self.output, feed_dict = {self.input : fullInput})
    
    def train(self, observations, actions, rewards, stateValues, gamma):
        inputs = np.concatenate((observations, actions), axis = -1)
        targets = rewards + gamma*stateValues
        self.global_step = self.global_step + 1
        self.sess.run(self.minimizationOperation, feed_dict = {self.target : targets, self.input : inputs}) 