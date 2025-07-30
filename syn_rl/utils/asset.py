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


# def compute_GAE(rewards, values, dones, gamma, lam):
#     T = rewards.shape[0]
#     advantages = torch.zeros_like(rewards)
#     last_gae = 0.0
    
#     for t in reversed(range(T)):
#         if t == T - 1:
#             next_value = 0.0  # or values[T] if you have it
#         else:
#             next_value = values[t + 1]
        
#         delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
#         advantages[t] = delta + gamma * lam * (1 - dones[t]) * last_gae
#         last_gae = advantages[t]
    
#     returns = advantages + values[:-1]  # or values[:T] depending on your setup
#     return advantages, returns

def compute_GAE(rewards, values, is_terminals, gamma, lam):
    advantages = []
    returns = []
    gae = 0
    # `values` is a list of length len(rewards) + 1
    for t in reversed(range(len(rewards))):
        mask = 0.0 if is_terminals[t] else 1.0
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
    adv_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
    ret_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
    # normalize advantages
    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
    return adv_tensor, ret_tensor

# def compute_GAE(rewards, values, dones, gamma, lam):
#     """
#     rewards     : list or 1D-tensor of length T
#     values      : list or 1D-tensor of length T+1
#     dones       : list or 1D-tensor of booleans of length T
#     gamma, lam  : scalars
#     returns
#       adv  : (T,) tensor of advantages
#       ret  : (T,) tensor of TD(λ) returns
#     """
#     T = len(rewards)
#     # make sure we’re not backpropagating through the value estimates
#     values = torch.as_tensor(values, dtype=torch.float32).detach().to(device)
#     rewards = torch.as_tensor(rewards, dtype=torch.float32).to(device)
#     dones   = torch.as_tensor(dones,   dtype=torch.bool).to(device)

#     adv     = torch.zeros(T, dtype=torch.float32)
#     last_gae = 0.0

#     for t in reversed(range(T)):
#         mask = 0.0 if dones[t] else 1.0
#         # δ_t = r_t + γ V_{t+1} * mask − V_t
#         delta = rewards[t] + gamma * values[t+1] * mask - values[t]
#         last_gae = delta + gamma * lam * mask * last_gae
#         adv[t] = last_gae

#     # compute the “λ‐returns”
#     ret = adv + values[:-1]

#     # optional normalization
#     adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

#     return adv, ret
