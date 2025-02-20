import torch

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

def map_to_range(action, range):
    min_val, max_val = range
    mapped_action = ((action + 1) / 2) * (max_val - min_val) + min_val
    return mapped_action

def np_to_torch(x):
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)

def torch_to_np(x):
    return x.squeeze(0).cpu().detach().numpy().ravel()
