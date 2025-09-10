import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def map_to_range(action, range):
    min_val, max_val = np.array(range)
    mapped_action = ((action + 1) / 2) * (max_val - min_val) + min_val
    return mapped_action


def np_to_torch(x):
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)


def torch_to_np(x):
    return x.squeeze(0).cpu().detach().numpy().ravel()


def compute_rewards_to_go(rewards, dones, gamma):
    returns = []
    running_return = 0
    rewards_flat = rewards.ravel()
    dones_flat = dones.ravel()
    # Iterate backwards over rewards and dones
    for reward, done in zip(reversed(rewards_flat), reversed(dones_flat)):
        if done:
            running_return = 0  # Reset at end of trajectory
        running_return = reward + gamma * running_return
        returns.insert(0, running_return)
    return torch.tensor(returns).reshape(rewards.shape)



def compute_GAE(rewards, state_values, next_state_values, dones, gamma=0.99, lam=0.95):
    # Append bootstrap value (0.0) for last step
    # zero_value = torch.zeros((1, 1), device=state_values.device, dtype=state_values.dtype)
    values = torch.cat([state_values.detach(), next_state_values[-1:].detach()], dim=0)  # [T+1, 1]

    T = rewards.size(0)
    advantages = torch.zeros((T, 1), dtype=torch.float32, device=rewards.device)
    gae = torch.zeros((1, 1), dtype=torch.float32, device=rewards.device)

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]  # 0 if done else 1
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns