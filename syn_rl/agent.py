import numpy as np
import gymnasium as gym

class RLAgent:
    def __init__(self, env, algorithm, num_episodes=1000, max_steps=1000):
        # extract environment parameters
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        # extract algortithm parameters
        self.algorithm = algorithm
        self.nn_layers =         

    def train(self):
        self.algorithm.train(self.env, self.num_episodes, self.max_steps)

    def evaluate(self, num_episodes=100):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    