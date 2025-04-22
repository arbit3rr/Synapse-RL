import numpy as np
import random
from collections import deque


class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size, return_all=False):
        # Sample a batch of experiences
        if return_all:
            sampled_experiences = self.buffer
        else:
            sampled_experiences = random.sample(self.buffer, batch_size)
        # Transpose the list of experiences, then convert each component to a NumPy array
        sampled_experiences = [np.array(x) for x in zip(*sampled_experiences)]
        # Ensure each component has at least 2 dimensions
        return [x if x.ndim >= 2 else np.expand_dims(x, axis=-1) for x in sampled_experiences]
    
    def compute_rewards_to_go(self, reward_idx, done_idx, gamma):
        returns = np.zeros(len(self.buffer))
        running_return = 0
        rewards = np.array([exp[reward_idx] for exp in self.buffer], dtype=np.float32)
        dones = np.array([exp[done_idx] for exp in self.buffer], dtype=np.float32)
        # Compute rewards-to-go in reverse
        for j in reversed(range(len(rewards))):
            if dones[j]:
                running_return = 0
            running_return = rewards[j] + gamma * running_return
            returns[j] = running_return
        # Append the computed returns to each experience
        for i in range(len(self.buffer)):
            self.buffer[i].append(returns[i])      # Append reward-to-go

    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        return self.buffer[idx]
    
    def clear(self):
        self.buffer.clear()


class RolloutBuffer():
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Sample a batch of experiences
        sampled_experiences = random.sample(self.buffer, batch_size)
        # Transpose the list of experiences, then convert each component to a NumPy array
        sampled_experiences = [np.array(x) for x in zip(*sampled_experiences)]
        # Ensure each component has at least 2 dimensions
        return [x if x.ndim >= 2 else np.expand_dims(x, axis=-1) for x in sampled_experiences]
    
    def compute_rewards_to_go(self, reward_idx, done_idx, gamma):
        returns = np.zeros(len(self.buffer))
        running_return = 0
        rewards = np.array([exp[reward_idx] for exp in self.buffer], dtype=np.float32)
        dones = np.array([exp[done_idx] for exp in self.buffer], dtype=np.float32)
        # Compute rewards-to-go in reverse
        for j in reversed(range(len(rewards))):
            if dones[j]:
                running_return = 0
            running_return = rewards[j] + gamma * running_return
            returns[j] = running_return
        # Append the computed returns to each experience
        for i in range(len(self.buffer)):
            self.buffer[i].append(returns[i])      # Append reward-to-go

    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        return self.buffer[idx]
    
    def clear(self):    
        self.buffer.clear()