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
        
    def sample(self, batch_size, return_next_state=False, return_all=False):
        # Sample a batch of experiences
        if return_all:
            sampled_experiences = self.buffer
        else:
            start_idx = random.randint(0, len(self.buffer)-batch_size-1)
            sampled_experiences = list(itertools.islice(self.buffer, start_idx, start_idx + batch_size))

        # Transpose the list of experiences, then convert each component to a NumPy array
        sampled_experiences = [np.array(x) for x in zip(*sampled_experiences)]

        # If include_next_state is True, fetch one additional experience for the next state
        if return_next_state:
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


# class RolloutBuffer:
#     def __init__(self, buffer_size):
#         self.buffer = deque(maxlen=buffer_size)

#     def push(self, experience):
#         self.buffer.append(experience)
        
#     def sample(self, batch_size, return_all=False, include_next_state=False):
#         # Sample a batch of experiences
#         if return_all or batch_size > len(self.buffer):
#             sampled_experiences = list(self.buffer)
#             start_idx = 0
#         else:
#             start_idx = random.randint(0, len(self.buffer) - batch_size)
#             sampled_experiences = list(itertools.islice(self.buffer, start_idx, start_idx + batch_size))
        
#         # If include_next_state is True, fetch one additional experience for the next state
#         extra_state = None
#         if include_next_state and start_idx + batch_size < len(self.buffer):
#             extra_experience = self.buffer[start_idx + batch_size]
#             extra_state = extra_experience[3]  # Assuming next_state is at index 3 (s, a, r, ns, d)

#         # Transpose the list of experiences, then convert each component to a NumPy array
#         sampled_experiences = [np.array(x) for x in zip(*sampled_experiences)]
        
#         # Ensure each component has at least 2 dimensions
#         sampled_experiences = [x if x.ndim >= 2 else np.expand_dims(x, axis=-1) for x in sampled_experiences]
        
#         # If extra state is requested, append it to the states (sampled_experiences[0])
#         if include_next_state and extra_state is not None:
#             # Expand extra_state to match the shape of states
#             extra_state = np.array(extra_state)
#             if extra_state.ndim < 2:
#                 extra_state = np.expand_dims(extra_state, axis=0)
#             # Append extra_state to the states array
#             sampled_experiences[0] = np.concatenate([sampled_experiences[0], extra_state], axis=0)
        
#         return sampled_experiences
        
#     def __len__(self):
#         return len(self.buffer)
    
#     def __getitem__(self, idx):
#         return self.buffer[idx]
    
#     def clear(self):    
#         self.buffer.clear()



    # def compute_rewards_to_go(self, reward_idx, done_idx, gamma):
    #     returns = np.zeros(len(self.buffer))
    #     running_return = 0
    #     rewards = np.array([exp[reward_idx] for exp in self.buffer], dtype=np.float32)
    #     dones = np.array([exp[done_idx] for exp in self.buffer], dtype=np.float32)
    #     # Compute rewards-to-go in reverse
    #     for j in reversed(range(len(rewards))):
    #         if dones[j]:
    #             running_return = 0
    #         running_return = rewards[j] + gamma * running_return
    #         returns[j] = running_return
    #     # Append the computed returns to each experience
    #     for i in range(len(self.buffer)):
    #         self.buffer[i].append(returns[i])      # Append reward-to-go
