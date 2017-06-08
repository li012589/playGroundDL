import tensorflow as tf
import gym
import numpy as np
from Qlearn import *

RENDER_ENV = False
ENV_NAME = 'CartPole-v0'
maxBuffSize = 10000
BATCH_SIZE = 64
SAVE_PER_STEP = 10000

OBSERVE = False
OBSERVE_TIME = 0
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.001 # starting value of epsilon
EXPLORE = 200000000

# Max training steps
MAX_EPISODES = 500000000000
# Max episode length
MAX_EP_STEPS = 1000
LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001

def train(sess,env,network):
    network.initTarget()
    Buff = replayBuff(maxBuffSize)
    #sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("savedQnetwork")
    epsilon = INITIAL_EPSILON
    t = 0
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    #for i in xrange(MAX_EPISODES):
    while True:
        s = env.reset()
        for j in xrange(MAX_EP_STEPS):
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
            if d:
                r = -1
            Buff.add(s,a,r,d,s2)
            if t > OBSERVE_TIME and Buff.size>BATCH_SIZE:
                s_batch, a_batch, r_batch, d_batch, s2_batch = Buff.sample(BATCH_SIZE)
                target_q = network.targetPredict(s2_batch)
                print(target_q)
                y_batch = []
                for k in xrange(BATCH_SIZE):
                    if d_batch[k]:
                        y_batch.append(r_batch[k])
                    else:
                        y_batch.append(r_batch[k]+GAMMA*np.max(target_q[k]))
                network.train(np.reshape(s_batch,(BATCH_SIZE,network.sDim)),np.reshape(a_batch,(BATCH_SIZE,network.aDim)),np.reshape(y_batch,(BATCH_SIZE,1)))
                network.targetUpdate()
            s = s2
            t += 1
            if t % SAVE_PER_STEP == 0:
                saver.save(sess, 'savedQnetwork/' + ENV_NAME + '-dqn', global_step = t)
            # print info
            state = ""
            if t <= OBSERVE_TIME:
                state = "observe"
            elif t > OBSERVE_TIME and t <= OBSERVE_TIME + EXPLORE:
                state = "explore"
            else:
                state = "train"
            #print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", a, "/ REWARD", r, "/ Q_MAX %e" % np.max(q))
            #print(aIndex)
            #print(a)
            if d:
                print("break")
                break
        #print("run out of steps")
    print ("run out of episodes")

def main():
    env = gym.make(ENV_NAME)
    sess = tf.InteractiveSession()
    sDim = env.observation_space.shape[0]
    aDim = 2#env.action_space.shape[0]
    network = Qnetwork(sess,sDim,aDim,LEARNING_RATE,TAU)
    train(sess,env,network)

if __name__ == "__main__":
    main()