import tensorflow as tf
from Qlearn import *
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

def showPic(x,y,z1,z2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, z1,color = 'r')
    ax.plot_trisurf(x, y, z2,color = 'b')
    #plt.show()
    return fig

def genPic(network,ranges,steps,choice,maxRange,basePath=-1,i=-1):
    x_dot = []
    theta_dot = []
    one_r = []
    zero_r = []
    x = 0
    theta = 0
    ranges = [ranges[t] for t in choice]
    batch = []
    for xDot in np.arange(ranges[0][0],ranges[0][1],steps[0]):
        for thetaDot in np.arange(ranges[1][0],ranges[1][1],steps[1]):
            #x_dot.append(xDot)
            #theta_dot.append(thetaDot)
            batch.append([x,xDot,theta,thetaDot])
            #q = network.predict(np.reshape([x,xDot,theta,thetaDot],(1,network.sDim)))[0]
            #zero_r.append(q[0])
            #one_r.append(q[1])
    #print len(batch)
    batch = np.reshape(batch,[len(batch),network.sDim])
    q = network.predict(batch)
    #print q.shape
    x_dot = [t[1] for t in batch]
    theta_dot = [t[3] for t in batch]
    one_r = [t[0] for t in q]
    zero_r = [t[1] for t in q]
    if basePath == -1:
        showPic(x_dot,theta_dot,zero_r,one_r)
    else:
        with open(basePath + "NNZeroresult" + str(i), "wb") as fp:
            pickle.dump(zero_r, fp)
        with open(basePath + "NNOneresult" + str(i), "wb") as fp:
            pickle.dump(one_r, fp)
        with open(basePath + "NNXresult" + str(i), "wb") as fp:
            pickle.dump(x_dot, fp)
        with open(basePath + "NNYresult" + str(i), "wb") as fp:
            pickle.dump(theta_dot, fp)

def save2Pic(basePath,ranges,step):
    for i in range(0,ranges,step):
        if i == 0:
            continue
        with open(basePath + "NNZeroresult" + str(i), "rb") as fp:
            zero_r = pickle.load(fp)
        with open(basePath + "NNOneresult" + str(i), "rb") as fp:
            one_r = pickle.load(fp)
        with open(basePath + "NNXresult" + str(i), "rb") as fp:
            x_dot = pickle.load(fp)
        with open(basePath + "NNYresult" + str(i), "rb") as fp:
            theta_dot = pickle.load(fp)
        fig = showPic(x_dot,theta_dot,zero_r,one_r)
        fig.savefig(basePath+str(i)+'.png')
        plt.close(fig)

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
    genPic(network,ranges,steps,[1,3],MAX_RANGE,BASE_DIR,0)
    save2Pic(BASE_DIR,1,1)
if __name__ == "__main__":
    ENV_NAME = 'CartPole-v0'
    LEARNING_RATE = 0.001
    TAU = 0.001
    STEP = [0.1,0.1,0.01,0.1]
    MAX_RANGE = 10
    BASE_DIR = './picDir/'
    main()