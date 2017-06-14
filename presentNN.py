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
        ranges.append([high[i],low[i]])
    #print(ranges)
    network = Qnetwork(sess,sDim,aDim,LEARNING_RATE,TAU)
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("savedQnetwork")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    x_dot = []
    theta_dot = []
    one_r = []
    zero_r = []
    x = 0
    theta = 0
    for xDot in np.arange(ranges[1][1],ranges[1][0],STEP[1]):
        for thetaDot in np.arange(ranges[3][1],ranges[3][0],STEP[3]):
            x_dot.append(xDot)
            theta_dot.append(thetaDot)
            q = network.predict(np.reshape([x,xDot,theta,thetaDot],(1,network.sDim)))[0]
            zero_r.append(q[0])
            one_r.append(q[1])
    #print x_dot
    #print theta_dot
    #print zero_r
    #print one_r
    #print(len(x_dot))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x_dot, theta_dot, one_r,color = 'r')
    #fig2 = plt.figure()
    #ax2 = fig2.add_subplot(111, projection='3d')
    ax.plot_trisurf(x_dot, theta_dot, zero_r,color = 'b')
    #Axes3D.plot_wireframe(x_dot, theta_dot, zero_r, rstride=10, cstride=10)
    plt.show()
if __name__ == "__main__":
    main()