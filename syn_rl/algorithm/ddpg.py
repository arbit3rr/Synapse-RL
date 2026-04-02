import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from ..network.policy import DeterministicPolicyNetwork
from ..network.value import QNetwork
from ..utils.asset import map_to_range, np_to_torch, torch_to_np
from ..utils.buffer import ExpBuffer
from ..utils.plot import plot_return
from ..utils.logger import TensorboardWriter

device = "cuda" if torch.cuda.is_available() else "cpu"
    
class DDPG:
    def __init__(self, state_size, action_size, action_range, hidden_dim=[128], gamma=0.99, min_uncertainty=0.1, uncertainty_decay=0.998, lr=3e-4, tau=0.005, buffer_size=1e5, batch_size=256):
        self.state_size = state_size
        self.action_size = action_size
        self.action_range = action_range
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.min_uncertainty = torch.tensor(min_uncertainty)
        self.uncertainty_decay = torch.tensor(uncertainty_decay)
        self.batch_size = batch_size
        self.memory = ExpBuffer(buffer_size)
        # actor (policy)
        self.actor = DeterministicPolicyNetwork(state_size, action_size, hidden_dim).to(device)
        self.target_actor = DeterministicPolicyNetwork(state_size, action_size, hidden_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        # critic (state-action value)
        self.critic = QNetwork(state_size, action_size, hidden_dim).to(device)
        self.target_critic = QNetwork(state_size, action_size, hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr, weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, weight_decay=1e-4)
        # log writer
        self.writer = TensorboardWriter(log_dir="Logs/DDPG")
        self.episode = 0
        self.iter = 0

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert data to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones).to(device)

        # Compute Q-Learning targets
        next_actions = self.target_actor(next_states)
        q_values_next = self.target_critic(next_states, next_actions)
        q_targets = rewards + (self.gamma * q_values_next * torch.logical_not(dones))
        
        # Compute Q-Learning loss and update the network parameters
        q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(q_values, q_targets.detach())
        
        # Update critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
        
        # Compute actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        # write loss values
        self.writer.log_scalar("Loss/Actor", actor_loss, self.iter)
        self.writer.log_scalar("Loss/Critic", critic_loss, self.iter)
        self.iter += 1

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)
    

    def decay_epsilon(self):
        # self.actor.uncertainty[self.actor.uncertainty > self.min_uncertainty] *= self.uncertainty_decay
        self.actor.uncertainty = torch.minimum(self.actor.uncertainty*self.uncertainty_decay, self.min_uncertainty)


    def evaluate(self, env):
        done, trunc = False, False
        score = 0
        state, _ = env.reset()
        while not (done or trunc):
            # Use the policy to select an action (without exploration)
            state_t = np_to_torch(state).to(device)
            action_t = self.actor.select_action(state_t)
            action = torch_to_np(action_t)
            mapped_action = map_to_range(action, self.action_range)
            next_state, reward, done, trunc, _ = env.step(mapped_action)
            state = next_state
            score += reward
        # Save best model
        if score > self.best_avg_reward:
            self.best_avg_reward = score
            self.save_checkpoint("Logs/DDPG_best_checkpoint.pth")
        # Log episode reward
        self.writer.log_scalar("Episode/Return Eval", score, self.episode)

    def train(self, env, episodes):
        if self.writer.run_dir is None:
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
                action_t = self.actor.select_action(state_t)
                # convert to numpy
                action = torch_to_np(action_t)
                # map action to range
                mapped_action = map_to_range(action, self.action_range)
                # take action
                next_state, reward, done, trunc, info = env.step(mapped_action)
                # store in memory
                self.memory.push([state, action, reward, next_state, done])
                # train agent
                self.learn()
                state = next_state
                score += reward
                length += 1
            # decrease exploration
            self.decay_epsilon()
            # log episode info
            self.writer.log_scalar("Episode/Return", score, self.episode)
            self.writer.log_scalar("Episode/Length", length, self.episode)
            # store episode return
            returns.append(score)
            plot_return(returns, f'Deep Deterministic Policy Gradient (DDPG) ({device})')
            # Evaluation
            if (self.episode + 1) % 20 == 0: self.evaluate(env)
            self.episode += 1
        env.close()
        self.writer.stop()
        return returns

    def save_checkpoint(self, filepath="Logs/DDPG_checkpoint.pth"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_uncertainty': self.actor.uncertainty,
            'episode': self.episode,
            'iter': self.iter,
            'best_avg_reward': self.best_avg_reward,
            'log_run_dir': self.writer.run_dir,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath="Logs/DDPG_checkpoint.pth"):
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor.uncertainty = checkpoint['actor_uncertainty']
        self.episode = checkpoint['episode']
        self.iter = checkpoint['iter']
        self.best_avg_reward = checkpoint['best_avg_reward']
        if 'log_run_dir' in checkpoint:
            self.writer.start(resume_dir=checkpoint['log_run_dir'])
        print(f"Checkpoint loaded from {filepath} (episode={self.episode}, iter={self.iter})")