import numpy as np
import random
from collections import deque
import itertools


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
        
    def sample(self, batch_size, include_next_state=False, return_all=False):
        # Sample a batch of experiences
        if return_all:
            sampled_experiences = self.buffer
        else:
            start_idx = random.randint(0, len(self.buffer)-batch_size-1)
            sampled_experiences = list(itertools.islice(self.buffer, start_idx, start_idx+batch_size))
        # Transpose the list of experiences, then convert each component to a NumPy array
        sampled_experiences = [np.array(x) for x in zip(*sampled_experiences)]
        # If include_next_state is True, fetch one additional experience for the next state
        if include_next_state:
            extra_experience = self.buffer[start_idx + batch_size]
            extra_state = extra_experience[0]  # Assuming next_state is at index 3 (s, a, r, ns, d)
            sampled_experiences[0] = np.concatenate([sampled_experiences[0], np.expand_dims(extra_state,0)], axis=0)
        # Ensure each component has at least 2 dimensions
        return [x if x.ndim >= 2 else np.expand_dims(x, axis=-1) for x in sampled_experiences]
        
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        return self.buffer[idx]
    
    def clear(self):    
        self.buffer.clear()

