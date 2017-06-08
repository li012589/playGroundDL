import tensorflow as tf
import numpy as np
from collections import deque
import random
import tflearn
import gym

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

class replayBuff:
    def __init__(self,maxSize):
        self.buffer = deque()
        self.size = 0
        self.maxSize = maxSize
    def add(self,s,a,r,t,s2):
        if self.size <= self.maxSize:
            self.buffer.append((s,a,r,t,s2))
            self.size += 1
        else:
            self.buffer.popleft()
            self.buffer.append((s,a,r,t,s2))
    def sample(self,batchSize):
        batch = []
        if self.size < batchSize:
            batch = random.sample(self.buffer, self.size)
        else:
            batch = random.sample(self.buffer,batchSize)
            sBatch = np.array([_[0] for _ in batch])
            aBatch = np.array([_[1] for _ in batch])
            rBatch = np.array([_[2] for _ in batch])
            tBatch = np.array([_[3] for _ in batch])
            s2Batch = np.array([_[4] for _ in batch])
        return sBatch, aBatch, rBatch, tBatch, s2Batch
    def clear(self):
        self.buffer.clear()
        self.size = 0

class Qnetwork:
    def __init__(self,sess,sDim,aDim,learningRate,tau):
        self.sess = sess
        self.sDim = sDim
        self.aDim = aDim
        self.learningRate = learningRate
        self.tau = tau

        self.inputs,self.out = self.createNetwork()
        self.trainableVar = tf.trainable_variables()

        self.targetInputs,self.targetOut = self.createNetwork()
        self.targetTrainableVar = tf.trainable_variables()[len(self.trainableVar):]

        self.updateTargetNetwork = [self.targetTrainableVar[i].assign(tf.multiply(self.trainableVar[i], self.tau)+tf.multiply(self.targetTrainableVar[i],1-self.tau)) for i in range(len(self.targetTrainableVar))]

        self.a = tf.placeholder("float", [None, 2])
        self.y = tf.placeholder("float", [None,1])
        self.readout_action = tf.reduce_sum(tf.multiply(self.out, self.a), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y - self.readout_action))
        self.train = tf.train.AdamOptimizer(1e-6).minimize(self.cost)
        self.sess.run(tf.global_variables_initializer())
    def createNetwork(self):
        W_fc1 = weight_variable([4,10])
        b_fc1 = bias_variable([10])
        W_fc2 = weight_variable([10,2])
        b_fc2 = bias_variable([2])
        #W_fc3 = weight_variable([400, self.aDim])
        #b_fc3 = bias_variable([self.aDim])
        inputs = tf.placeholder(tf.float32, [None,4])
        h_fc1 = tf.nn.relu(tf.matmul(inputs, W_fc1) + b_fc1)
        #h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        #h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
        outputs = tf.matmul(h_fc1,W_fc2) + b_fc2 #see ddpg for details in init w between -0.003--0.003
        return inputs,outputs
    def targetUpdate(self):
        return self.sess.run(self.updateTargetNetwork)

if __name__ =="__main__":
    sess = tf.InteractiveSession()
    ENV_NAME = 'CartPole-v0'
    env = gym.make(ENV_NAME)
    s_t = env.reset()
    Buff = replayBuff(10000)
    net = Qnetwork(sess,4,2,0.01,0.5)
    net.targetUpdate()

    D = deque()
    t = 0
    while True:
        if t >= 1000:
            pass
            #break
        t += 1
        q = net.out.eval(feed_dict={net.inputs : [s_t]})
        a_t = np.zeros([2])
        action_index = 0
        if random.random() <= 0.001:
            print("----------Random Action----------")
            action_index = random.randrange(2)
            a_t[random.randrange(2)] = 1
        else:
            action_index = np.argmax(q)
            a_t[action_index] = 1
        s_t1,r_t,terminal,_ = env.step(action_index)
        D.append((s_t, a_t, r_t, s_t1, terminal))
        s_t = s_t1
        if terminal:
            s_t = env.reset()
            print("break")
        if len(D) > 10000:
            D.popleft()
        if t > 10:
            minibatch = random.sample(D, 10)
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]
            y_batch = []
            readout_j1_batch = net.targetOut.eval(feed_dict = {net.targetInputs : s_j1_batch})
            print(readout_j1_batch)
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + 0.99 * np.max(readout_j1_batch[i]))
            #print(y_batch)
            #print(len(s_j_batch))
            #print(len(a_batch))
            #print(len(r_batch))
            #print(len(s_j1_batch))
            net.train.run(feed_dict = {net.y : np.reshape(y_batch,(10,1)),net.a : np.reshape(a_batch,(10,2)),net.inputs : np.reshape(s_j_batch,(10,4))})
            net.targetUpdate()
