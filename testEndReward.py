import tensorflow as tf
import gym
import numpy as np
from Qlearn import *
from gym import wrappers
from presentNN import genPic,save2Pic
import math

RENDER_ENV = False
ENV_NAME = 'CartPole-v0'
SAVE_PATH = './cartpole_MC'
maxBuffSize = 10000
BATCH_SIZE = 64
SAVE_PER_STEP = 1000

OBSERVE = False
NN_PRESENT = True
OBSERVE_TIME = 0
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.001 # starting value of epsilon
EXPLORE = 200000000

MAX_EPISODES = 10000
# Max episode length
MAX_EP_STEPS = 1000
LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001
STEP = [0.1,0.1]
MAX_RANGE = 10

def train(sess,env,network):
    sess.run(tf.global_variables_initializer())
    #network.initTarget()
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("savedMCNN")
    epsilon = INITIAL_EPSILON
    Buff = replayBuff(maxBuffSize)
    t = 0
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    for i in xrange(MAX_EPISODES):
        #print i
        s = env.reset()
        reward = 0
        n = 0
        for j in xrange(MAX_EP_STEPS):
            n += 1
            t += 1
            if RENDER_ENV:
                env.render()
            if random.random() <= epsilon:
                #print("exploring")
                aIndex = env.action_space.sample()
            else:
                q = network.predict(np.reshape(s,(1,network.sDim)))[0]
                #print (q)
                aIndex = np.argmax(q)

            a = np.zeros([network.aDim])
            a[aIndex] = 1
            if epsilon > FINAL_EPSILON and t > OBSERVE_TIME:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            s2,r,d,info = env.step(aIndex)
            reward += r
            Buff.add(s,a,r,d,s2)
            print("TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", a, "/ REWARD", r, "/ Q_MAX %e" % np.max(q))
            if d:
                if reward != 200:
                    rr = -1*n
                else:
                    rr = 200
                s_batch, a_batch, _, d_batch, s2_batch = Buff.sample(n)
                Buff.clear()
                target_q = network.predict(s2_batch)
                y_batch = [rr/n  for _ in range(n)]
                network.train(np.reshape(s_batch,(n,network.sDim)),np.reshape(a_batch,(n,network.aDim)),np.reshape(y_batch,(n,1)))
                #network.targetUpdate()
                print("Done")
                break
            s = s2

def main():
    env = gym.make(ENV_NAME)
    env = wrappers.Monitor(env, SAVE_PATH)
    sess = tf.InteractiveSession()
    sDim = env.observation_space.shape[0]
    aDim = env.action_space.n
    #high = env.observation_space.high
    #low = env.observation_space.low
    network = Qnetwork(sess,sDim,aDim,LEARNING_RATE,TAU)
    train(sess,env,network)

if __name__ == "__main__":
    main()
