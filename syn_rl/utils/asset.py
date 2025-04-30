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



# def compute_GAE(rewards, values, dones, gamma, lam):
#     T, D = rewards.shape
#     device = rewards.device
#     advantages = torch.zeros_like(rewards, device=device)
#     last_gae = torch.zeros(1, D, device=device)
    
#     for t in reversed(range(T)):
#         mask = 1.0 - dones[t]  # [1, D], zeroes out if done
#         delta = (
#             rewards[t]
#             + gamma * values[t + 1] * mask
#             - values[t]
#         )  # [1, D]
#         last_gae = delta + gamma * lam * mask * last_gae
#         advantages[t] = last_gae
#     # vlues has one extra value at the end (for the last state)
#     returns = advantages + values[:-1]
#     return advantages, returns


def compute_GAE(rewards, values, dones, gamma, lam):
    T, D = rewards.shape
    device = rewards.device
    
    # Compute delta for all time steps
    mask = 1.0 - dones  # [T, D]
    delta = rewards + gamma * values[1:] * mask - values[:-1]  # [T, D]
    
    # Reverse for forward recurrence
    delta_rev = delta.flip(0)  # [T, D]
    c_rev = (gamma * lam * mask).flip(0)  # [T, D]
    
    # Initialize output
    advantages_rev = torch.zeros_like(rewards, device=device)  # [T, D]
    advantages_rev[0] = delta_rev[0]
    
    # Compute forward recurrence (still requires a loop in pure PyTorch)
    for t in range(1, T):
        advantages_rev[t] = delta_rev[t] + c_rev[t-1] * advantages_rev[t-1]
    
    # Reverse back to original order
    advantages = advantages_rev.flip(0)
    returns = advantages + values[:-1]
    return advantages, returns


