import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.nn import GaussianPolicyNetwork, ValueNetwork
from utils.asset import map_to_range, np_to_torch, torch_to_np
from utils.asset import compute_rewards_to_go
from utils.buffer import ReplayBuffer
from utils.plot import plot_return
from utils.logger import TensorboardWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

class PPOAgent():
    def __init__(self, state_size, action_size, action_range, hidden_dim=[128], gamma=0.99, lr=3e-4, buffer_size=1e5):
        self.state_size = state_size
        self.action_size = action_size
        self.action_range = action_range
        self.gamma = gamma
        self.lr = lr
        self.buffer_size = int(buffer_size)
        self.memory = ReplayBuffer(self.buffer_size)
        self.clip_ratio = 0.2

        # Actor (policy)
        self.new_policy = GaussianPolicyNetwork(state_size, action_size, hidden_dim).to(device)
        self.old_policy = GaussianPolicyNetwork(state_size, action_size, hidden_dim).to(device)
        self.old_policy.load_state_dict(self.new_policy.state_dict())

        # Critic (state value)
        self.value_network = ValueNetwork(state_size, hidden_dim).to(device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.new_policy.parameters(), lr=self.lr, weight_decay=1e-4)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.lr, weight_decay=1e-4)

        # Log writer
        self.writer = TensorboardWriter(log_dir="Logs/PPO", comment="PPO")
        self.iter = 0

    def learn(self):
        if len(self.memory) == 0:
            return  # Avoid training if no data is available
        
        # Read from replay buffer
        states, old_log_probs, rewards, dones = self.memory.sample(None, return_all=True)

        # Convert data to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32).to(device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        # Compute Value Targets
        discounted_returns = compute_rewards_to_go(rewards, self.gamma)
        state_values = self.value_network(states)

        # Compute Advantage and Normalize
        advantages = discounted_returns - state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        # Compute Value Loss
        value_loss = F.mse_loss(discounted_returns, state_values)

        # Update Value Network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Compute new log probs
        _, action_log_probs, entropy = self.new_policy.select_action(states, return_entropy=True)
        ratios = torch.exp(action_log_probs - old_log_probs.detach())

        # PPO Clipped Objective
        surr1 = ratios * advantages.detach()
        surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages.detach()
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy regularization
        entropy_coef = 0.01  # Adjust this value to control exploration
        entropy_loss = -entropy.mean()  # We maximize entropy, so we take negative
        total_policy_loss = policy_loss + entropy_coef * entropy_loss

        # Update Actor Network
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()

       # Update Old Policy
        self.old_policy.load_state_dict(self.new_policy.state_dict())

        # write loss values
        self.writer.log_scalar("Loss/Policy", policy_loss, self.iter)
        self.writer.log_scalar("Loss/Entropy", entropy_loss, self.iter)
        self.writer.log_scalar("Loss/Value", value_loss, self.iter)
        self.iter += 1

        # clear memory
        self.memory.clear()

    def train(self, env, episodes):
        returns = []
        for episode in range(episodes):
            score = 0
            length = 0
            done = False
            state, _ = env.reset()
            while not done:
                # convert to tensor
                state_t = np_to_torch(state).to(device)
                # select action
                action_t, action_log_prob_t = self.old_policy.select_action(state_t)
                # convert to numpy
                action = torch_to_np(action_t)
                action_log_prob = torch_to_np(action_log_prob_t)
                # map action to range
                mapped_action = map_to_range(action, self.action_range)
                # take action
                next_state, reward, done, _, info = env.step(mapped_action)
                # store in memory
                self.memory.push([state, action_log_prob, reward, done])
                state = next_state
                score += reward
                length += 1
            # train agent
            self.learn()
            # log episode info
            self.writer.log_scalar("Episode/Return", score, episode)
            self.writer.log_scalar("Episode/Length", length, episode)
            # store episode return
            returns.append(score)
            plot_return(returns, f'Proximal Policy Optimization (PPO) ({device})')

        env.close()
        self.writer.close()
        return returns
