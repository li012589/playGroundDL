import tensorflow as tf
import gym
import numpy as np
from Qlearn import *

RENDER_ENV = True
ENV_NAME = 'CartPole-v0'
maxBuffSize = 10000
BATCH_SIZE = 64
SAVE_PER_STEP = 10000

OBSERVE = False
OBSERVE_TIME = 100000
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
EXPLORE = 200000000

# Max training steps
MAX_EPISODES = 50000
# Max episode length
MAX_EP_STEPS = 1000
LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001

def train(sess,env,network):
    network.updateTargetNetwork()
    Buff = replayBuff(maxBuffSize)
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("savedQnetwork")
    epsilon = INITIAL_EPSILON
    t = 0
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    for i in xrange(MAX_EPISODES):
        s = env.reset()
        for j in xrange(MAX_EP_STEPS):
            if RENDER_ENV:
                env.render()
            if random.random() <= epsilon:
                a = env.action_space.sample()
            else:
                a = np.argmax(network.predict(np.reshape(s,(1,network.sDim)))[0])

            if epsilon > FINAL_EPSILON and t > OBSERVE_TIME:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            s2,r,t,info = env.step(a)
            Buff.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,t, np.reshape(s2, (actor.s_dim,)))
            if t > OBSERVE_TIME and Buff.size()>BATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(BATCH_SIZE)
                target_q = critic.predict_target(s2_batch)
                y_batch = []
                for k in xrange(BATCH_SIZE):
                    if t_batch[k]:
                        y_batch.append(r_batch[k])
                    else:
                        y_batch.append(r_batch[k]+GAMMA*target_q[k])
                network.train(s_batch,a_batch,np.reshape(y_batch,(BATCH_SIZE,1)))
                network.updateTargetNetwork()
            s = s2
            t += 1
            if t % SAVE_PER_STEP == 0:
            saver.save(sess, 'savedQnetwork/' + GAME + '-dqn', global_step = t)
            if t:
                break

def main():
    env = gym.make(ENV_NAME)
    sess = tf.InteractiveSession()
    sDim = env.observation_space.shape[0]
    aDim = env.action_space.shape[0]
    network = Qnetwork(sess,sDim,aDim,LEARNING_RATE,TAU)
    train(sess,env,network)

if __name__ == "__main__":
    main()