import tensorflow as tf
import numpy as np
from collections import deque
import random
import tflearn
import gym

USE_RFLEARN = False
USE_TARGET_NETWORK = True

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

        self.a = tf.placeholder(tf.float32, [None, self.aDim])
        self.y = tf.placeholder(tf.float32,[None,1])

        self.predictionValue = tf.reduce_sum(tf.multiply(self.out, self.a), reduction_indices=1)
        self.tmp = tf.transpose(self.y) - self.predictionValue
        #print(tuple(self.tmp.get_shape().as_list()))
        self.tmp2 = tf.square(self.tmp)
        self.loss = tf.reduce_mean(self.tmp2)

        self.optimize = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
    def createNetwork(self):
        W_fc1 = weight_variable([self.sDim,10])
        b_fc1 = bias_variable([10])
        W_fc2 = weight_variable([10, self.aDim])
        b_fc2 = bias_variable([self.aDim])
        #W_fc3 = weight_variable([400, self.aDim])
        #b_fc3 = bias_variable([self.aDim])
        inputs = tf.placeholder(tf.float32, [None,self.sDim])
        h_fc1 = tf.nn.relu(tf.matmul(inputs, W_fc1) + b_fc1)
        #h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        #h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
        out = tf.matmul(h_fc1,W_fc2) + b_fc2 #see ddpg for details in init w between -0.003--0.003
        return inputs,out
    def train(self,inputs,actions,y):
        return self.sess.run(self.optimize,feed_dict={self.inputs:inputs,self.a:actions,self.y:y})
    def predict(self,inputs):
        return self.sess.run(self.out,feed_dict={self.inputs:inputs})
    def targetPredict(self,inputs):
        return self.sess.run(self.targetOut,feed_dict={self.targetInputs:inputs})
    def targetUpdate(self):
        return self.sess.run(self.updateTargetNetwork)

if __name__ == "__main__":
    ENV_NAME = 'CartPole-v0'
    Buff = replayBuff(10000)
    env = gym.make(ENV_NAME)
    s = env.reset()
    x = np.random.rand(1000,2)
    yt = np.asarray([t[0]**2+t[1]**2 for t in x])
    sess = tf.InteractiveSession()
    network = Qnetwork(sess,4,2,0.01,1)
    network.targetUpdate()
    at = np.ones(1)
    savedloss = []
    #for t in range(len(x)):
    for i in range(1,100):
        q = network.predict(np.reshape(s,(1,network.sDim)))[0]
        #print(q)
        aIndex = np.argmax(q)
        a = np.zeros([network.aDim])
        a[aIndex] = 1
        #print(a)
        s2,r,d,_ = env.step(aIndex)
        Buff.add(np.reshape(s, (network.sDim,)), np.reshape(a, (network.aDim,)), r,d, np.reshape(s2, (network.sDim,)))
        #pre = network.predict(np.reshape(x[t],(1,network.sDim)))#sess.run(out,feed_dict={inputs:np.reshape(x[t],(1,2))})
        if Buff.size>10:
            s_batch, a_batch, r_batch, d_batch, s2_batch = Buff.sample(10)
            #print(s_batch)
            #print(a_batch)
            #print(r_batch)
            #print(d_batch)
            #print(s2_batch)
            target_q = network.targetPredict(s2_batch)
            print(target_q)
            y_batch = []
            for k in xrange(10):
                if d_batch[k]:
                    y_batch.append(r_batch[k])
                else:
                    y_batch.append(r_batch[k]+0.99*np.max(target_q[k]))
            #print(target_q)
            #print(sess.run(network.predictionValue,feed_dict={network.inputs:s_batch,network.a:a_batch}))
            network.train(s_batch,a_batch,np.reshape(y_batch,(10,1)))
            network.targetUpdate()
        #print pre
        #print t
        #pre = sess.run(out,feed_dict={inputs:np.reshape(x[t],(1,2))})
        #los = sess.run(network.loss,feed_dict={network.inputs:np.reshape(x[t],(1,2)),network.a:np.reshape(at,(1,1)),network.y:np.reshape(yt[t],(1,1))})
        #print(sess.run(network.loss,feed_dict={network.inputs:np.reshape(x[t],(1,2)),network.a:np.reshape(at,(1,1)),network.y:np.reshape(yt[t],(1,1))}))
        #print los
        #savedloss.append(los)
        #print yt
        #network.train(np.reshape(x[t],(1,2)),np.reshape(at,(1,1)),np.reshape(yt[t],(1,1)))
    #print(sum(savedloss)/float(len(savedloss)))
    #print(sess.run(W_fc1))
    #print(sess.run(W_fc2))
    #print(sess.run(W_fc3))