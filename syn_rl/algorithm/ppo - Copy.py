import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from ..network.policy import GaussianPolicyNetwork
from ..network.value import ValueNetwork
from ..utils.asset import map_to_range, np_to_torch, torch_to_np, compute_GAE
from ..utils.buffer import ExpBuffer
from ..utils.plot import plot_return
from ..utils.logger import TensorboardWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

class PPO:
    def __init__(self, state_size, action_size, action_range, hidden_dim=[128], 
                 gamma=0.99, lr=3e-4, clip_ratio=0.2, buffer_size=500, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.action_range = action_range
        self.gamma = gamma
        self.lr = lr
        self.clip_ratio = clip_ratio
        self.batch_size = batch_size
        self.memory = ExpBuffer(buffer_size)
        self.buffer_size = buffer_size
        self.learn_freq = 10 

        # Actor (policy)
        self.policy = GaussianPolicyNetwork(state_size, action_size, hidden_dim).to(device)
        self.policy_old = GaussianPolicyNetwork(state_size, action_size, hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Critic (state value)
        self.value = ValueNetwork(state_size, hidden_dim).to(device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, weight_decay=1e-4)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=3*self.lr, weight_decay=1e-4)

        # Log writer
        self.writer = TensorboardWriter(log_dir="Logs/PPO", comment="PPO")
        self.iter = 0
        self.best_avg_reward = -np.inf
        
    def learn(self):
        if len(self.memory) <= self.batch_size:
            return
        
        # Read from replay buffer
        states, actions, old_log_probs, rewards, dones = self.memory.sample(self.batch_size, include_next_state=True)

        # Convert data to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        
        # Obtain value estimates
        state_values = self.value(states).detach()

        # Compute GAE advantages and returns
        advantages = []
        returns = []
        gae = 0
        lam = 0.95
        values = list(state_values) + [torch.tensor(0.0).to(device)]
        for t in reversed(range(len(rewards))):
            mask = 0 if dones[t] else 1
            delta = rewards[t] + self.gamma * values[t+1] * mask - values[t]
            gae = delta + self.gamma * lam * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(-1).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # advantages, discounted_returns = compute_GAE(rewards, state_values, dones, self.gamma, lam=0.95)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute Value Loss
        state_values = self.value(states[:-1])
        value_loss = F.mse_loss(returns, state_values)

        # Update Value Network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Compute new log probs
        action_log_probs, entropy = self.policy.evaluate(states[:-1], actions)
        ratios = torch.exp(action_log_probs - old_log_probs)

        # PPO Clipped Objective
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.clip_ratio, 1+self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy regularization
        entropy_coef = 0.01  # Adjust this value to control exploration
        entropy_loss = -entropy.mean()  # We maximize entropy, so we take negative
        total_policy_loss = policy_loss + entropy_coef * entropy_loss

        # Update Actor Network
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()

        # write loss values
        self.writer.log_scalar("Loss/Policy", policy_loss, self.iter)
        self.writer.log_scalar("Loss/Entropy", entropy_loss, self.iter)
        self.writer.log_scalar("Loss/Value", value_loss, self.iter)
        self.iter += 1


    def train(self, env, episodes):
        returns = []
        counter = 0
        for episode in range(episodes):
            score = 0
            length = 0
            done, trunc = False, False
            state, _ = env.reset()
            while not (done or trunc):
                # convert to tensor
                state_t = np_to_torch(state).to(device)
                # select action
                action_t, action_log_prob_t = self.policy_old.select_action(state_t)
                # convert to numpy
                action = torch_to_np(action_t)
                action_log_prob = torch_to_np(action_log_prob_t)
                # map action to range
                mapped_action = map_to_range(action, self.action_range)
                # take action
                next_state, reward, done, trunc, info = env.step(mapped_action)
                # store in memory
                self.memory.push([state, action, action_log_prob, reward, done])
                # train agent
                if counter % self.batch_size == 0:
                    self.learn()
                state = next_state
                score += reward
                length += 1
                counter += 1
            # update old policy
            if self.iter % self.learn_freq == 0:
                self.policy_old.load_state_dict(self.policy.state_dict())
                print("updated the old policy")
            # self.memory.clear()
            # log episode info
            self.writer.log_scalar("Episode/Return", score, episode)
            self.writer.log_scalar("Episode/Length", length, episode)
            # store episode return
            returns.append(score)
            plot_return(returns, f'Proximal Policy Optimization (PPO) ({device})')

        env.close()
        self.writer.close()
        return returns
