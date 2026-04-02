import os
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

class PPO_EP:
    def __init__(self, state_size, action_size, action_range, hidden_dim=[128], 
                 gamma=0.99, lam=0.95, lr=3e-4, clip_ratio=0.1, K_epochs=100, buffer_size=2e3):
        self.state_size = state_size
        self.action_size = action_size
        self.action_range = action_range
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.clip_ratio = clip_ratio
        self.K_epochs = K_epochs
        self.buffer_size = buffer_size
        self.memory = ExpBuffer(buffer_size)

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
        self.writer = TensorboardWriter(log_dir="Logs/PPO_EP")
        self.iter = 0
        self.best_avg_reward = -np.inf
        
    def learn(self):
        # Read from replay buffer
        states, actions, old_log_probs, rewards, next_states, dones = self.memory.sample(batch_size=np.inf, return_all=True)
        # states, actions, old_log_probs, rewards, dones = zip(*self.memory.buffer)

        # Convert data to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        # Obtain value estimates
        state_values = self.value(states)
        next_state_values = self.value(next_states)

        # Compute GAE advantages and returns
        advantages, returns = compute_GAE(rewards, state_values, next_state_values, dones, self.gamma, lam=0.95)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimize
        for _ in range(self.K_epochs):
            # Compute Value Loss
            state_values = self.value(states)
            value_loss = F.mse_loss(state_values, returns)

            # Update Value Network
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            # Compute new log probs
            action_log_probs, entropy = self.policy.evaluate(states, actions)
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

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear()

        # write loss values
        self.writer.log_scalar("Loss/Policy", policy_loss, self.iter)
        self.writer.log_scalar("Loss/Entropy", entropy_loss, self.iter)
        self.writer.log_scalar("Loss/Value", value_loss, self.iter)
        self.iter += 1


    def train(self, env, episodes):
        if self.writer.run_dir is None:
            self.writer.start()
        returns = []
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
                self.memory.push([state, action, action_log_prob, reward, next_state, done])
                # train agent
                if len(self.memory.buffer) >= self.buffer_size:
                    self.learn()
                state = next_state
                score += reward
                length += 1
            # log episode info
            self.writer.log_scalar("Episode/Return", score, episode)
            self.writer.log_scalar("Episode/Length", length, episode)
            # store episode return
            returns.append(score)
            plot_return(returns, f'Proximal Policy Optimization (PPO) ({device})')

        env.close()
        self.writer.stop()
        return returns

    def save_checkpoint(self, filepath="Logs/PPO_EP_checkpoint.pth"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'policy_old_state_dict': self.policy_old.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'iter': self.iter,
            'best_avg_reward': self.best_avg_reward,
            'log_run_dir': self.writer.run_dir,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath="Logs/PPO_EP_checkpoint.pth"):
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.iter = checkpoint['iter']
        self.best_avg_reward = checkpoint['best_avg_reward']
        if 'log_run_dir' in checkpoint:
            self.writer.start(resume_dir=checkpoint['log_run_dir'])
        print(f"Checkpoint loaded from {filepath} (iter={self.iter})")
