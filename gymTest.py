import tensorflow as tf
import gym
import numpy as np
from Qlearn import *

RENDER_ENV = True
ENV_NAME = 'CartPole-v0'
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64


def main():
    env = gym.make(ENV_NAME)
    sDim = env.observation_space.shape[0]
    aDim = env.action_space.shape[0]

if __name__ == "__main__":
    main()