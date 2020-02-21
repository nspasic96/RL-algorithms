import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions 

dev = tf.Variable(23.1, "std")
dev2 = tf.Variable(1., "std2")

dist = tfd.Normal(loc=0., scale=dev)
dist2 = tfd.Normal(loc=0., scale=dev2)

sess = tf.Session()

batch_size = 20
iters = 100000

targets = tf.placeholder(shape=[batch_size], dtype=tf.float32, name="targets")

inputs = dist.sample(sample_shape = [batch_size])
loss = tf.losses.mean_squared_error(targets, inputs)
minimizeOp = tf.train.AdamOptimizer().minimize(loss = loss)

inputs2 = dist2.sample(sample_shape = [batch_size])
loss2 = tf.losses.mean_squared_error(targets, inputs2)
minimizeOp2 = tf.train.AdamOptimizer().minimize(loss = loss2)

init_op1 = tf.initialize_all_variables()
init_op2 = tf.initialize_local_variables()
sess.run(init_op1)
sess.run(init_op2)

print(sess.run(dist.log_prob(0.)))
print(sess.run(dist2.log_prob([-1.,1])))

samplesBefore = sess.run(dist.sample([10]))
print("Samples before: {}\n".format(samplesBefore))
samplesBefore = sess.run(dist2.sample([10]))
print("Samples before2: {}\n".format(samplesBefore))

for i in range(iters):
    x = np.random.randn(batch_size)
    sess.run(minimizeOp, feed_dict = {targets : x})
    sess.run(minimizeOp2, feed_dict = {targets : x})

samplesAfter, stddev, stddev2 = sess.run([dist.sample([10]), dev, dev2])
print("Samples after: {}\n".format(samplesAfter))
print("Stddev: {}\n".format(stddev))
print("Stddev2: {}\n".format(stddev2))
