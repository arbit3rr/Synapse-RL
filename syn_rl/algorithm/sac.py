import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from ..network.policy import GaussianPolicyNetwork
from ..network.value import QNetwork
from ..utils.asset import map_to_range, np_to_torch, torch_to_np
from ..utils.buffer import ExpBuffer
from ..utils.plot import plot_return
from ..utils.logger import TensorboardWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

class SAC:
    def __init__(self, state_size, action_size, action_range, hidden_dim=[128], 
                 alpha=0.1, gamma=0.99, lr=3e-4, tau=0.005, buffer_size=1e5, batch_size=256):
        self.state_size = state_size
        self.action_size = action_size
        self.action_range = action_range
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.memory = ExpBuffer(buffer_size)

        # Trainable entropy temperature
        self.target_entropy = -action_size  # A heuristic for continuous action spaces
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)

        # Actor (policy)
        self.actor = GaussianPolicyNetwork(state_size, action_size, hidden_dim).to(device)

        # Critics
        self.QNet1 = QNetwork(state_size, action_size, hidden_dim).to(device)
        self.target_QNet1 = QNetwork(state_size, action_size, hidden_dim).to(device)
        self.target_QNet1.load_state_dict(self.QNet1.state_dict())

        self.QNet2 = QNetwork(state_size, action_size, hidden_dim).to(device)
        self.target_QNet2 = QNetwork(state_size, action_size, hidden_dim).to(device)
        self.target_QNet2.load_state_dict(self.QNet2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr, weight_decay=1e-4)
        self.QNet1_optimizer = optim.Adam(self.QNet1.parameters(), lr=self.lr, weight_decay=1e-4)
        self.QNet2_optimizer = optim.Adam(self.QNet2.parameters(), lr=self.lr, weight_decay=1e-4)

        # Logging
        self.writer = TensorboardWriter(log_dir="Logs/SAC")
        self.episode = 0
        self.iter = 0
        self.best_avg_reward = -np.inf

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample from replay buffer
        states, actions, action_log_probs, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert data to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        action_log_probs = torch.tensor(action_log_probs, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        # Compute Q targets
        with torch.no_grad():
            alpha = self.log_alpha.exp()  # Convert log_alpha to alpha
            next_actions, next_action_log_probs = self.actor.select_action(next_states)
            q_values_next = torch.min(self.target_QNet1(next_states, next_actions), 
                                      self.target_QNet2(next_states, next_actions)) - alpha * next_action_log_probs
            q_targets = rewards + (self.gamma * q_values_next * (1 - dones))
            q_targets = q_targets.detach()

        # Update Q1
        q1_values = self.QNet1(states, actions)
        Q1_loss = F.mse_loss(q1_values, q_targets)
        self.QNet1_optimizer.zero_grad()
        Q1_loss.backward()
        self.QNet1_optimizer.step()

        # Update Q2
        q2_values = self.QNet2(states, actions)
        Q2_loss = F.mse_loss(q2_values, q_targets)
        self.QNet2_optimizer.zero_grad()
        Q2_loss.backward()
        self.QNet2_optimizer.step()

        # Compute actor loss
        actions, action_log_probs = self.actor.select_action(states)
        actor_loss = -(torch.min(self.QNet1(states, actions), self.QNet2(states, actions)) - alpha * action_log_probs).mean()

        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha (train entropy)
        alpha_loss = -self.log_alpha * (action_log_probs + self.target_entropy).detach().mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Log loss values
        self.writer.log_scalar("Loss/Actor", actor_loss, self.iter)
        self.writer.log_scalar("Loss/Q1", Q1_loss, self.iter)
        self.writer.log_scalar("Loss/Q2", Q2_loss, self.iter)
        self.writer.log_scalar("Entropy/Alpha_Loss", alpha_loss, self.iter)
        self.writer.log_scalar("Entropy/Alpha", alpha, self.iter)
        self.iter += 1

        # Soft update of target networks
        for target_param, param in zip(self.target_QNet1.parameters(), self.QNet1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_QNet2.parameters(), self.QNet2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def evaluate(self, env):
        done, trunc = False, False
        score = 0
        state, _ = env.reset()
        while not (done or trunc):
            # Use the policy to select an action (without exploration)
            state_t = np_to_torch(state).to(device)
            action_t, _ = self.actor.select_action(state_t, deterministic=True)
            action = torch_to_np(action_t)
            mapped_action = map_to_range(action, self.action_range)
            next_state, reward, done, trunc, info = env.step(mapped_action)
            state = next_state
            score += reward
        # Save best model
        if score > self.best_avg_reward:
            self.best_avg_reward = score
            self.save_checkpoint("Logs/SAC_best_checkpoint.pth")
        # Log episode reward
        self.writer.log_scalar("Episode/Return Eval", score, self.episode)

    def train(self, env, episodes):
        self.writer.start()
        returns = []
        for _ in range(episodes):
            score = 0
            length = 0
            done, trunc = False, False
            state, _ = env.reset()
            while not (done or trunc):
                # convert to tensor
                state_t = np_to_torch(state).to(device)
                # select action
                action_t, action_log_prob_t = self.actor.select_action(state_t)
                # convert to numpy
                action = torch_to_np(action_t)
                action_log_prob = torch_to_np(action_log_prob_t)
                # map action to range
                mapped_action = map_to_range(action, self.action_range)
                # take action
                next_state, reward, done, trunc, info = env.step(mapped_action)
                # store in memory
                self.memory.push([state, action, action_log_prob, reward, next_state,  done])
                # train agent
                self.learn()
                state = next_state
                score += reward
                length += 1
            # log episode info
            self.writer.log_scalar("Episode/Return", score, self.episode)
            self.writer.log_scalar("Episode/Length", length, self.episode)
            # store episode return
            returns.append(score)
            plot_return(returns, f'Soft Actor Critic (SAC) ({device})')
            # Evaluation
            if (self.episode + 1) % 20 == 0: self.evaluate(env)
            self.episode += 1
        env.close()
        self.writer.stop()
        return returns

    def save_checkpoint(self, filepath="Logs/SAC_checkpoint.pth"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'QNet1_state_dict': self.QNet1.state_dict(),
            'QNet2_state_dict': self.QNet2.state_dict(),
            'target_QNet1_state_dict': self.target_QNet1.state_dict(),
            'target_QNet2_state_dict': self.target_QNet2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'QNet1_optimizer_state_dict': self.QNet1_optimizer.state_dict(),
            'QNet2_optimizer_state_dict': self.QNet2_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'episode': self.episode,
            'iter': self.iter,
            'best_avg_reward': self.best_avg_reward,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath="Logs/SAC_checkpoint.pth"):
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.QNet1.load_state_dict(checkpoint['QNet1_state_dict'])
        self.QNet2.load_state_dict(checkpoint['QNet2_state_dict'])
        self.target_QNet1.load_state_dict(checkpoint['target_QNet1_state_dict'])
        self.target_QNet2.load_state_dict(checkpoint['target_QNet2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.QNet1_optimizer.load_state_dict(checkpoint['QNet1_optimizer_state_dict'])
        self.QNet2_optimizer.load_state_dict(checkpoint['QNet2_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha'].to(device).requires_grad_(True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.episode = checkpoint['episode']
        self.iter = checkpoint['iter']
        self.best_avg_reward = checkpoint['best_avg_reward']
        print(f"Checkpoint loaded from {filepath} (episode={self.episode}, iter={self.iter})")
