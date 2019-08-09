import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, Flatten, Activation
from keras.optimizers import Adam

def create_network():
    fc1 = Dense(50, activation = Activation('relu'))
    outputs = Dense(3)

    model = Sequential()
    model.add(fc1)
    model.add(outputs)
    model.compile(optimizer = Adam(0.003), loss ="mse")
    return model
    
model1 = create_network()
model2 = create_network()

randInp = np.random.rand(1,50)
gt1 = model1.predict(randInp)
gt2 = model2.predict(randInp)

model2.set_weights(model1.get_weights())
v1 = model2.predict(randInp)
model2.fit(randInp, np.random.rand(1,3))
v2 = model1.predict(randInp)
v3 = model2.predict(randInp)

print("Should be v1==gt1==v2!=v3\n gt1={}\n gt2={}\n v1={} \n v2={}\n v3={}".format(gt1,gt2,v1,v2,v3))


