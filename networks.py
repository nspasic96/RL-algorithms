import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions 


tfDtype = tf.float32 
class StateValueNetwork:
    
    def __init__(self, sess, inputShape, learningRate, suffix=""):
        self.inputShape = inputShape
        self.learningRate = learningRate
        self.sess = sess
        self.global_step = tf.Variable(0,dtype = tf.int32)
        self.i = 0
        self.initialized = False
        self.suffix = suffix
        
    def _createDefault(self):
        with tf.variable_scope("StateValueNetwork{}".format(self.suffix)):
            self.input = tf.placeholder(dtype = tfDtype, shape = [None, self.inputShape], name="input")
            curNode = tf.layers.Dense(64, tf.nn.relu, use_bias = True, kernel_initializer= tf.initializers.truncated_normal(), name="fc1")(self.input)
            self.output = tf.layers.Dense(1, use_bias = True, kernel_initializer= tf.initializers.truncated_normal(), name="output")(curNode)
            
            self.target = tf.placeholder(dtype = tfDtype, shape = [None, 1], name="target")
            self.loss = tf.losses.mean_squared_error(self.target, self.output)
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learningRate)
            
            #self.minimizationOperation = self.optimizer.minimize(self.loss, global_step = self.global_step)
            #self.gradientNorm = self.optimizer.compute_gradients(self.loss)            
                        
            gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, .5)
            self.minimizationOperation = self.optimizer.apply_gradients(zip(gradients, variables))
                    
    def forward(self, observations):
        if not self.initialized:
            self._createDefault()
            init = tf.initialize_local_variables()
            init2 = tf.initialize_all_variables()
            self.sess.run([init,init2])
            self.initialized = True
        
        assert (len(observations.shape) == 2 and observations.shape[1] == self.inputShape)
        return self.sess.run(self.output, feed_dict = {self.input : observations})
    
    def train(self, observations, Qs, logPis): 
        if not self.initialized:
            self._createDefault()
            init = tf.initialize_local_variables()
            init2 = tf.initialize_all_variables()
            self.sess.run([init,init2])
            self.initialized = True
            
        targets = Qs - logPis  
        self.global_step = self.global_step + 1

        self.sess.run(self.minimizationOperation, feed_dict = {self.target : targets, self.input : observations})

    def update(self, other, tau):
        #TODO: assign moving exponential weights to this network
        return 0
    
    def weightsAssign(self, other):
        return 0
        
class PolicyNetworkDiscrete:
    
    def __init__(self, sess, inputShape, outputShape):
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.sess = sess
        self.global_step = tf.Variable(0,dtype = tf.int32)
        self.i = 0
        self.initialized = False
        
        
    def _createDefault(self):
        with tf.variable_scope("PolicyNetworkDiscrete"):
            
            self.inputs = tf.placeholder(dtype = tfDtype, shape = [None, self.inputShape], name="observations")              
            curNode = tf.layers.Dense(120, tf.nn.relu, use_bias = True,  name="fc1")(self.inputs)
            curNode = tf.layers.Dense(84, tf.nn.relu, use_bias = True,  name="fc2")(curNode)
            self.actions = tf.layers.Dense(self.outputShape, use_bias = True,  name="actions")(curNode)
            self.logProbs = tf.nn.log_softmax(self.actions)
            
            
    def getSampledActions(self, inputs):
        if not self.initialized:
            self._createDefault()
            init = tf.initialize_local_variables()
            init2 = tf.initialize_all_variables()
            self.sess.run([init,init2])
            self.initialized = True
          
        actions = tf.squeeze(tf.random.categorical(self.logProbs, 1), axis=1)
        probs = tf.reduce_sum(tf.one_hot(actions)*self.logProbs)
            
        return self.sess.run([actions,probs], feed_dict = {self.inputs : inputs})
    
        
class PolicyNetwork:
    
    def __init__(self, sess, inputShape, outputShape, learningRate, squash):
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.learningRate = learningRate
        self.sess = sess
        self.global_step = tf.Variable(0,dtype = tf.int32)
        self.i = 0
        self.initialized = False
        self.squash = squash
        
    def _createDefault(self):
        with tf.variable_scope("PolicyNetwork"):
            self.inputs = tf.placeholder(dtype = tfDtype, shape = [None, self.inputShape], name="input")
            curNode = tf.layers.Dense(120, tf.nn.relu, use_bias = True,  name="fc1")(self.inputs)
            curNode = tf.layers.Dense(84, tf.nn.relu, use_bias = True,  name="fc2")(curNode)
            self.actionMean = tf.layers.Dense(self.outputShape, use_bias = True,  name="ActionsMean")(curNode)
            self.actionLogStd = tf.layers.Dense(self.outputShape, use_bias = True, name="ActionsLogStd")(curNode)
        
            self.actionLogStdCliped = tf.clip_by_value(self.actionLogStd, -20, 2)
            self.actionStd = tf.math.exp(self.actionLogStdCliped)
            
            self.actionSampler = tfd.Normal(loc=self.actionMean, scale=self.actionStd)
            self.actionRaw = self.actionSampler.sample()  
            
            if self.squash: 
                self.actionSquashed = tf.tanh(self.actionRaw)
            else:
                self.actionSquashed = self.actionRaw
            
            self.actionsPlh = tf.placeholder(dtype = tfDtype, shape = [None, 3])
            self.targets = tf.placeholder(dtype = tfDtype, shape = [None, 1])
            self.logProbs = tf.expand_dims(tf.reduce_sum(self.actionSampler.log_prob(self.actionsPlh), axis=-1),1)
            self.loss = tf.losses.mean_squared_error(self.targets, self.logProbs)
            
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learningRate)
            gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, .5)
            self.minimizationOperation = self.optimizer.apply_gradients(zip(gradients, variables))
            #self.minimizationOperation = tf.train.AdamOptimizer(learning_rate = self.learningRate).minimize(self.loss, global_step = self.global_step)
                           
    def getSampledAction(self, observations):
        if not self.initialized:
            self._createDefault()
            init = tf.initialize_local_variables()
            init2 = tf.initialize_all_variables()
            self.sess.run([init,init2])
            self.initialized = True
        return self.sess.run([self.actionSquashed, self.actionRaw, self.actionMean, self.actionStd], feed_dict = {self.inputs : observations})
        
    def train(self, observations, actions, Qs):
        
        targets = Qs       
        
        self.global_step = self.global_step + 1
        return self.sess.run(self.minimizationOperation, feed_dict = {self.inputs : observations, self.targets : targets, self.actionsPlh : actions})        
           
        
class SoftQNetwork:
    
    def __init__(self, sess, inputShape, outputShape, learningRate, alpha, suffix):
        self.inputShape = inputShape
        self.outputShape = outputShape
        self.learningRate = learningRate
        self.sess = sess
        self.alpha = alpha
        self.global_step = tf.Variable(0,dtype = tf.int32)
        self.i = 0
        self.initialized = False
        self.suffix = suffix
        
    def _createDefault(self):
        with tf.variable_scope("SoftQNetwork{}".format(self.suffix)):
            self.input = tf.placeholder(dtype = tfDtype, shape = [None, self.inputShape + self.outputShape], name="input")
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
        
    
    
        
        