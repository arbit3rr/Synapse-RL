import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


# Value Network architecture
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dims):
        super().__init__()
        # Build hidden layers from the list of hidden dimensions
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            input_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer for the value
        self.fc_out = nn.Linear(input_dim, 1)
    
    def forward(self, state):
        x = self.hidden_layers(state)
        value = self.fc_out(x)
        return value


# Q-Network architecture
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        # Build hidden layers from the list of hidden dimensions
        layers = []
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            input_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer for Q-value
        self.fc_out = nn.Linear(input_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.hidden_layers(x)
        q_value = self.fc_out(x)
        return q_value
    

# Deep Q-Network architecture (DQN)
class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, epsilon):
        super().__init__()
        # Build hidden layers from the list of hidden dimensions
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            input_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*layers)

        # Output layer for action
        self.fc_out = nn.Linear(input_dim, action_dim)
        self.action_dim = action_dim
        self.epsilon = epsilon

    def forward(self, state):
        x = self.hidden_layers(state)
        q_values = self.fc_out(x)
        return q_values
    
    def select_action(self, state):
        if  torch.rand(1) <= self.epsilon:
            return torch.randint(self.action_dim, (1,))
        q_values = self(state)
        return torch.argmax(q_values)
