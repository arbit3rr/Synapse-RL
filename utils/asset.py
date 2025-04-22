import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def map_to_range(action, range):
    min_val, max_val = range
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



def compute_GAE(rewards, values, dones, gamma, lam):
    T, D = rewards.shape
    device = rewards.device
    last_value = torch.zeros(1, 1).to(device)

    values_pad = torch.cat([values, last_value], dim=0)
    advantages = torch.zeros_like(rewards, device=device)
    last_gae = torch.zeros(1, D, device=device)

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]  # [1, D], zeroes out if done
        delta = (
            rewards[t]
            + gamma * values_pad[t + 1] * mask
            - values_pad[t]
        )  # [1, D]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


