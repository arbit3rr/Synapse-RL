import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


# Deterministic Policy Network architecture
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
    

# Gaussian Policy Network architecture
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

    def forward(self, state):
        x = self.hidden_layers(state)
        action_mean = self.fc_mean(x)
        action_log_std = torch.clamp(self.fc_log_std(x), min=-10, max=2)  # Adjusted range
        action_std = torch.exp(action_log_std)
        return action_mean, action_std

    def select_action(self, state, deterministic=False):
        mean, std = self(state)
        if deterministic: 
            action = torch.tanh(mean)  # Directly apply tanh for deterministic mode
            log_prob = torch.zeros_like(mean).sum(dim=-1, keepdim=True)
        else:
            normal_dist = torch.distributions.Normal(mean, std)
            action_pre_tanh = normal_dist.rsample()  # Sample before applying tanh
            log_prob = normal_dist.log_prob(action_pre_tanh).sum(dim=-1, keepdim=True)
            # Apply tanh transformation correctly
            action = torch.tanh(action_pre_tanh)
            # Log probability correction for tanh squashing
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)                
        return action, log_prob
    
    def evaluate(self, state, action):
        mean, std = self(state)
        normal_dist = torch.distributions.Normal(mean, std)
        # Since actions are after tanh, compute pre-tanh values
        action_clipped = torch.clamp(action, -0.999, 0.999)  # Clip for numerical stability
        action_pre_tanh = torch.atanh(action_clipped)
        # Compute log probability of pre-tanh actions
        log_prob = normal_dist.log_prob(action_pre_tanh).sum(dim=-1, keepdim=True)
        # Adjust for tanh transformation
        log_prob -= torch.log(1 - action_clipped.pow(2) + 1e-6).sum(dim=-1, keepdim=True)                
        # Compute entropy
        entropy = normal_dist.entropy().sum(dim=-1)
        return log_prob, entropy


# Categorical Policy Network architecture
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