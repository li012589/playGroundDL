import tensorflow as tf
from Qlearn import *
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ENV_NAME = 'CartPole-v0'
LEARNING_RATE = 0.001
TAU = 0.001
STEP = [0.1,0.1,0.01,0.1]
MAX_RANGE = 10

def genPic(network,ranges,steps,choice,maxRange):
    x_dot = []
    theta_dot = []
    one_r = []
    zero_r = []
    x = 0
    theta = 0
    ranges = [ranges[t] for t in choice]
    for xDot in np.arange(ranges[0][0],ranges[0][1],STEP[0]):
        for thetaDot in np.arange(ranges[1][0],ranges[1][1],STEP[1]):
            x_dot.append(xDot)
            theta_dot.append(thetaDot)
            q = network.predict(np.reshape([x,xDot,theta,thetaDot],(1,network.sDim)))[0]
            zero_r.append(q[0])
            one_r.append(q[1])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x_dot, theta_dot, one_r,color = 'r')
    ax.plot_trisurf(x_dot, theta_dot, zero_r,color = 'b')
    plt.show()

def main():
    env = gym.make(ENV_NAME)
    sDim = env.observation_space.shape[0]
    aDim = env.action_space.n
    sess = tf.InteractiveSession()
    high = env.observation_space.high
    low = env.observation_space.low
    ranges = []
    for i in range(len(high)):
        if high[i] >= MAX_RANGE:
            high[i] = MAX_RANGE
            low[i] = -MAX_RANGE
        ranges.append([low[i],high[i]])
    #print(ranges)
    network = Qnetwork(sess,sDim,aDim,LEARNING_RATE,TAU)
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("savedQnetwork")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    steps = [STEP[t] for t in [1,3]]
    genPic(network,ranges,steps,[1,3],MAX_RANGE)
if __name__ == "__main__":
    main()