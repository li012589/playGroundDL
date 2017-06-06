import tensorflow as tf
import numpy as np
from collections import deque
import random
import tflearn

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
        self.loss = tf.reduce_mean(tf.square(self.y - self.predictionValue))

        self.optimize = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)
    def createNetwork(self):
        W_fc1 = weight_variable([self.sDim,300])
        b_fc1 = bias_variable([300])
        W_fc2 = weight_variable([300, 400])
        b_fc2 = bias_variable([400])
        W_fc3 = weight_variable([400, self.aDim])
        b_fc3 = bias_variable([self.aDim])
        inputs = tf.placeholder(tf.float32, [None,self.sDim])
        h_fc1 = tf.nn.relu(tf.matmul(inputs, W_fc1) + b_fc1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        #h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
        out = tf.matmul(h_fc2,W_fc3) + W_fc3 #see ddpg for details in init w between -0.003--0.003
        self.sess.run(tf.global_variables_initializer())
        return inputs,out
    def train(self,inputs,actions,y):
        return self.sess.run(self.optimize,feed_dict={self.inputs:inputs,self.a:actions,self.y:y})
    def predict(self,inputs):
        return self.sess.run(self.out,feed_dict={self.inputs:inputs})
    def targetPredict(self,inputs):
        return self.sess.run(self.out,feed_dict={self.targetInputs:inputs})
    def targetUpdate(self):
        return self.sess.run(self.updateTargetNetwork)

def trainQnetwork(sess,env,network,maxBuffSize,MAX_EPISODES,MAX_EP_STEPS,RENDER_ENV,INITIAL_EPSILON,FINAL_EPSILON,OBSERVE_TIME,EXPLORE,SAVE_PER_STEP,BATCH_SIZE):
    network.targetUpdate()
    Buff = replayBuff(maxBuffSize)
    #sess.run(tf.initialize_all_variables())
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
                network.targetUpdate()
            s = s2
            t += 1
            if t % SAVE_PER_STEP == 0:
                saver.save(sess, 'savedQnetwork/' + GAME + '-dqn', global_step = t)
            if t:
                break


if __name__ == "__main__":
    pass