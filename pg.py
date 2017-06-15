import numpy as np
import os
import sys

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rl"

import gym
from gym import wrappers

import tensorflow as tf
import pickle

reset_graph()

n_inputs = 4
n_hidden = 4
n_outputs = 1

learning_rate = 0.01

initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs)
outputs = tf.nn.sigmoid(logits)  # probability of action 0 (left)
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

y = 1. - tf.to_float(action)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]
gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

env = gym.make("CartPole-v0")
env = wrappers.Monitor(env, './cartpole')

n_games_per_update = 10
n_max_steps = 1000
n_iterations = 30
save_iterations = 1
NN_PRESENT = True
discount_rate = 0.95
record = []
basePath = './picDirPG/'
MAX_RANGE = 10
with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        print("\rIteration: {}".format(iteration))
        all_rewards = []
        all_gradients = []
        for game in range(n_games_per_update):
            current_rewards = []
            current_gradients = []
            obs = env.reset()
            for step in range(n_max_steps):
                action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs.reshape(1, n_inputs)})
                if iteration == n_iterations-1:
                    record.append(action_val[0][0])
                    env.render()
                obs, reward, done, info = env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate=discount_rate)
        feed_dict = {}
        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                          for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients
        sess.run(training_op, feed_dict=feed_dict)
        if iteration % save_iterations == 0:
            if NN_PRESENT:
                high = env.observation_space.high
                low = env.observation_space.low
                ranges = []
                for i in range(len(high)):
                    if high[i] >= MAX_RANGE:
                        high[i] = MAX_RANGE
                        low[i] = -MAX_RANGE
                    ranges.append([low[i],high[i]])
                    #genPic(network,ranges,STEP,[1,3],MAX_RANGE,BASE_DIR_PIC,t)
                x = 0
                ranges = [ranges[t] for t in [1,2,3]]
                steps = [0.1,0.1,0.1]
                batch = []
                for xDot in np.arange(ranges[0][0],ranges[0][1],steps[0]):
                    for theta in np.arange(ranges[1][0],ranges[1][1],steps[1]):
                        for thetaDot in np.arange(ranges[2][0],ranges[2][1],steps[2]):
                            #x_dot.append(xDot)
                            #theta_dot.append(thetaDot)
                            batch.append([x,xDot,theta,thetaDot])
                batch = np.reshape(batch,[len(batch),n_inputs])
                actions = sess.run(action, feed_dict={X: batch})
                x_dot = [t[1] for t in batch]
                theta_dot = [t[3] for t in batch]
                theta = [t[2] for t in batch]
                onesX = [ (x_dot[i]) for i in range(len(actions)) if actions[i] == 1 ]
                onesY = [ (theta_dot[i]) for i in range(len(actions)) if actions[i] == 1 ]
                onesZ = [ (theta[i]) for i in range(len(actions)) if actions[i] == 1 ]
                ones = [onesX,onesZ,onesZ]
                zeroX = [ (x_dot[i]) for i in range(len(actions)) if actions[i] == 0 ]
                zeroY = [ (theta_dot[i]) for i in range(len(actions)) if actions[i] == 0 ]
                zeroZ = [ (theta[i]) for i in range(len(actions)) if actions[i] == 0 ]
                zeros = [zeroX,zeroY,zeroZ]
                with open(basePath + "NNZeroresult" + str(iteration), "wb") as fp:
                    pickle.dump(zeros, fp)
                with open(basePath + "NNOneresult" + str(iteration), "wb") as fp:
                    pickle.dump(ones, fp)
            #saver.save(sess, "./my_policy_net_pg.ckpt")
env.reset()
#print record
#i = 0
#for val in record:
#    i+=1
#    #env.render()
#    env.step(val)
#    if done:
#        break
#env.close()
#print i
#print len(record)
#frames = render_policy_net("./my_policy_net_pg.ckpt", action, X, n_max_steps=1000)
#video = plot_animation(frames)
#plt.show()

