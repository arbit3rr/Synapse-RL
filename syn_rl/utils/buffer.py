import numpy as np
import random
from collections import deque
import itertools


class ExpBuffer():
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=int(buffer_size))

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size, return_all=False, keep_order=False):
        # Sample a batch of experiences
        if return_all:
            sampled_experiences = self.buffer
        elif keep_order:
            start_idx = random.randint(0, len(self.buffer)-batch_size)
            sampled_experiences = list(itertools.islice(self.buffer, start_idx, start_idx+batch_size))
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


