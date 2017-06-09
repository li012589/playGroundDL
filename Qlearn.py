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
        self.initTargetNetwork = [self.targetTrainableVar[i].assign(self.trainableVar[i]) for i in range(len(self.targetTrainableVar))]

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
        W_fc1 = weight_variable([self.sDim,50])
        b_fc1 = bias_variable([50])
        W_fc2 = weight_variable([50, 40])
        b_fc2 = bias_variable([40])
        W_fc3 = weight_variable([40, self.aDim])
        b_fc3 = bias_variable([self.aDim])
        inputs = tf.placeholder(tf.float32, [None,self.sDim])
        h_fc1 = tf.nn.relu(tf.matmul(inputs, W_fc1) + b_fc1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        #h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
        out = tf.matmul(h_fc2,W_fc3) + b_fc3 #see ddpg for details in init w between -0.003--0.003
        return inputs,out
    def train(self,inputs,actions,y):
        return self.sess.run(self.optimize,feed_dict={self.inputs:inputs,self.a:actions,self.y:y})
    def predict(self,inputs):
        return self.sess.run(self.out,feed_dict={self.inputs:inputs})
    def targetPredict(self,inputs):
        return self.sess.run(self.targetOut,feed_dict={self.targetInputs:inputs})
    def targetUpdate(self):
        return self.sess.run(self.updateTargetNetwork)
    def initTarget(self):
        return self.sess.run(self.initTargetNetwork)

if __name__ == "__main__":
   pass
   


