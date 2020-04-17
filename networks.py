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
            
            curNode = tf.layers.Dense(self.hiddenLayers[0], tf.nn.tanh, name="fc1")(self.input)
            for i,l in enumerate(self.hiddenLayers[1:]):
                curNode = tf.layers.Dense(l, tf.nn.tanh,  name="fc{}".format(i+2))(curNode)
            
            self.output = tf.layers.Dense(1, name="output")(curNode)
            
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
    
    def __init__(self, sess, inputLength, outputLength, hiddenLaySizes, hiddenLayerActivations, inputPh, actionPh, attachActionLayer, suffix, reuse=None):
        self.inputLength = inputLength
        self.outputLength = outputLength
        self.hiddenLayers = hiddenLaySizes
        self.hiddenLayerActivations = hiddenLayerActivations
        self.sess = sess
        self.global_step = tf.Variable(0,dtype = tf.int32)
        self.i = 0
        self.suffix = suffix
        self.input = inputPh
        self.action = actionPh
        self.attachActionLayer = attachActionLayer #to which hidden layer to attach action(0 based indexed)
        self.reuse = reuse
        if len(hiddenLaySizes) <= attachActionLayer:
            print("\nattachActionLayer={} outside of network({} hidden layers)\n".format(attachActionLayer, len(hiddenLayerActivations))) 
        self._createDefault()

    def _createDefault(self):
        if self.reuse is not None:
            with tf.variable_scope("QNetwork{}".format(self.reuse.suffix), reuse = True):
                curNode = tf.layers.Dense(self.hiddenLayers[0], self.hiddenLayerActivations[0], name="fc1")(self.input)
                if(self.attachActionLayer == 0):
                    curNode = tf.concat([curNode, self.action], axis = 1)
                for i,l in enumerate(self.hiddenLayers[1:]):
                    curNode = tf.layers.Dense(l, self.hiddenLayerActivations[i+1],  name="fc{}".format(i+2))(curNode)
                    if(self.attachActionLayer == i+1):
                        curNode = tf.concat([curNode, self.action], axis = 1, name="QNetworkActionConcat")
                
                self.output = tf.squeeze(tf.layers.Dense(1, self.hiddenLayerActivations[-1], name="output")(curNode),axis=1)
                self.variablesScope = "QNetwork{}".format(self.reuse.suffix) 
        else:   
            with tf.variable_scope("QNetwork{}".format(self.suffix)): 
                curNode = tf.layers.Dense(self.hiddenLayers[0], self.hiddenLayerActivations[0], name="fc1")(self.input)
                if(self.attachActionLayer == 0):
                    curNode = tf.concat([curNode, self.action], axis = 1)
                for i,l in enumerate(self.hiddenLayers[1:]):
                    curNode = tf.layers.Dense(l, self.hiddenLayerActivations[i+1], name="fc{}".format(i+2))(curNode)
                    if(self.attachActionLayer == i+1):
                        curNode = tf.concat([curNode, self.action], axis = 1, name="QNetworkActionConcat")
                
                self.output = tf.squeeze(tf.layers.Dense(1, self.hiddenLayerActivations[-1], name="output")(curNode),axis=1)
                self.variablesScope = "QNetwork{}".format(self.suffix) 
                   
    def forward(self, observations):        
        assert (len(observations.shape) == 2 and observations.shape[1] == self.inputLength)
        
        return self.sess.run(self.output, feed_dict = {self.input : observations})
           
class PolicyNetworkDiscrete:
    
    def __init__(self, sess, inputLength, outputLength, hiddenLaySizes, inputsPh, actionsPh, suffix):
        self.inputLength = inputLength
        self.outputLength = outputLength
        self.hiddenLayers = hiddenLaySizes
        self.sess = sess
        self.global_step = tf.Variable(0,dtype = tf.int32)
        self.i = 0
        self.inputs = inputsPh
        self.actions = actionsPh
        self.suffix = suffix
        self._createDefault()      
                
    def _createDefault(self):
        with tf.variable_scope("PolicyNetworkDiscrete{}".format(self.suffix)):
            
            curNode = tf.layers.Dense(self.hiddenLayers[0], tf.nn.tanh,  name="fc1")(self.inputs)
            for i,l in enumerate(self.hiddenLayers[1:]):
                curNode = tf.layers.Dense(l, tf.nn.tanh,  name="fc{}".format(i+2))(curNode)
            self.logits = tf.layers.Dense(self.outputLength,  name="actions")(curNode)
            self.logProbs = tf.nn.log_softmax(self.logits)
            
            self.sampledActions = tf.squeeze(tf.random.categorical(self.logProbs, 1), axis=1)
            self.sampledLogProbs = tf.reduce_sum(tf.one_hot(self.sampledActions, depth = self.outputLength)*self.logProbs)
            
            self.logProbWithCurrParams = tf.reduce_sum(tf.one_hot(tf.squeeze(self.actions,1), depth=self.outputLength)*self.logProbs, axis=1)#log probs for actions given the observation(both fed with placeholder)
            
            
    def getSampledActions(self, inputs):                      
        return self.sess.run([self.sampledActions, self.sampledLogProbs, self.logProbs], feed_dict = {self.inputs : inputs})


class PolicyNetworkContinuous:
    
    def __init__(self, sess, inputLength, outputLength, hiddenLaySizes, hiddenLayerActivations, inputPh, actionsPh, suffix, actionMeanScale=None, logStdInit=None, logStdTrainable=True, clipLogStd=None, actionClip=None, squashAction=False):
        self.inputLength = inputLength
        self.outputLength = outputLength
        self.input = inputPh
        self.actions = actionsPh
        self.hiddenLayers = hiddenLaySizes
        self.hiddenLayerActivations = hiddenLayerActivations
        self.sess = sess
        self.global_step = tf.Variable(0,dtype = tf.int32)
        self.i = 0
        self.clipLogStd = clipLogStd
        self.actionClip = actionClip
        self.actionMeanScale = actionMeanScale
        self.logStdInit = logStdInit
        #this applies only if logStds in not None
        self.logStdTrainable = logStdTrainable
        self.squashAction = squashAction
        self.suffix = suffix
        self.variablesScope = "PolicyNetworkContinuous{}".format(suffix)
        self._createDefault()
        
    def _createDefault(self):
        with tf.variable_scope("PolicyNetworkContinuous{}".format(self.suffix)):
            curNode = tf.layers.Dense(self.hiddenLayers[0], self.hiddenLayerActivations[0],  name="fc1")(self.input)
            for i,l in enumerate(self.hiddenLayers[1:]):
                curNode = tf.layers.Dense(l, self.hiddenLayerActivations[i+1],  name="fc{}".format(i+2))(curNode)
            self.actionMean = tf.layers.Dense(self.outputLength, self.hiddenLayerActivations[-1],  name="ActionsMean")(curNode)
            if(self.actionMeanScale is not None):
                assert(self.actionMeanScale.shape == (1,self.outputLength))
                self.actionMean = self.actionMean * self.actionMeanScale

            if self.logStdInit is not None:
                #print("self.logStdInit = {}. equal NaN = {}".format(self.logStdInit, self.logStdInit == -float("infinity")))
                if self.logStdInit != -float("infinity"):
                    assert(self.logStdInit.shape == (1,self.outputLength)) 
                    self.actionLogStd = tf.get_variable(name="ActionsLogStdDetached{}Trainable".format("" if self.logStdTrainable else "Non"), initializer=self.logStdInit, trainable=self.logStdTrainable)
                else:
                    self.actionLogStd = None
            else:
                self.actionLogStd = tf.layers.Dense(self.outputLength, name="ActionsLogStd")(curNode)
        
            if self.clipLogStd is not None:
                self.actionLogStd = tf.clip_by_value(self.actionLogStd, self.clipLogStd[0], self.clipLogStd[1], name="ClipedActionsLogStd")

            if self.actionLogStd is not None:   
                self.actionStd = tf.math.exp(self.actionLogStd)                
                self.actionRaw = self.actionMean + tf.random_normal(tf.shape(self.actionMean)) * self.actionStd
            else:
                self.actionRaw = self.actionMean

            
            #TODO: Check whether this work when squash=True(because gaussian_likelihood doesnt take it into consideration)
            if self.squashAction: 
                self.actionFinal = tf.tanh(self.actionRaw)
            else:
                self.actionFinal = self.actionRaw   
            
            if self.actionClip is not None: 
                assert(self.actionClip.shape == (2, self.outputLength) )
                self.actionFinal = tf.clip_by_value(self.actionFinal, self.actionClip[0,:], self.actionClip[1,:])

            if self.actionLogStd is not None:    
                self.sampledLogProbs = utils.gaussian_likelihood(self.actionRaw, self.actionMean, self.actionLogStd)
                self.logProbWithCurrParams = utils.gaussian_likelihood(self.actions, self.actionMean, self.actionLogStd)#log prob(joint, all action components are from gaussian) for action given the observation(both fed with placeholder)
            
                           
    def getSampledActions(self, observations):
        if self.actionLogStd is not None:
            return self.sess.run([self.actionFinal, self.sampledLogProbs, self.actionMean, self.actionLogStd], feed_dict = {self.input : observations})
        else:
            return self.sess.run([self.actionFinal, self.actionMean], feed_dict = {self.input : observations})

         
        
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
            curNode = tf.layers.Dense(120, tf.nn.relu, kernel_initializer= tf.initializers.truncated_normal(), name="fc1")(self.input)
            curNode = tf.layers.Dense(84, tf.nn.relu, kernel_initializer= tf.initializers.truncated_normal(), name="fc2")(curNode)
            self.output = tf.layers.Dense(1, kernel_initializer= tf.initializers.truncated_normal(), name="output")(curNode)
            
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