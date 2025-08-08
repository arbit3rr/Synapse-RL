import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, TanhTransform

device = "cuda" if torch.cuda.is_available() else "cpu"


# Deterministic Policy Network
class DeterministicPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        # Build hidden layers from the list of hidden dimensions
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            input_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*layers)

        # Output layers for action
        self.fc_out = nn.Linear(input_dim, action_dim)
        self.uncertainty = torch.ones(1).to(device)
        self.action_dim = action_dim

    def forward(self, state):
        x = self.hidden_layers(state)
        action = torch.tanh(self.fc_out(x))
        return action
    
    def select_action(self, state):
        action = self(state)
        return action + torch.randn(self.action_dim).to(device)*self.uncertainty


# Gaussian Policy Network
class GaussianPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        # Build hidden layers
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            input_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*layers)
        # Output layers for mean and standard deviation
        self.fc_mean = nn.Linear(input_dim, action_dim)
        self.fc_log_std = nn.Linear(input_dim, action_dim)
        nn.init.constant_(self.fc_log_std.bias, -0.3)  # log_std ≈ 0.6

    def forward(self, state):
        x = self.hidden_layers(state)
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, min=-10, max=2)
        std = torch.exp(log_std)
        return mean, std

    def _build_dist(self, mean, std):
        base = Normal(mean, std)
        # Apply tanh transform with caching
        transforms = [TanhTransform(cache_size=1)]
        return TransformedDistribution(base, transforms)
    
    def select_action(self, state, deterministic=False):
        mean, std = self(state)
        dist = self._build_dist(mean, std)
        if deterministic: 
            action = torch.tanh(mean)
        else: 
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob
    
    def evaluate(self, state, action):
        mean, std = self(state)
        dist = self._build_dist(mean, std)
        action = torch.clamp(action, -0.999, 0.999)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.base_dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy

    
# Categorical Policy Network
class CategoricalPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        # Build hidden layers
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            input_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*layers)
    
        # Output layers for mean and standard deviation
        self.fc_out = nn.Linear(input_dim, action_dim)

    def forward(self, state):
        x = self.hidden_layers(state)
        logits = self.fc_out(x)
        return F.softmax(logits, dim=-1)
    
    def select_action(self, state):
        probs = self(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)