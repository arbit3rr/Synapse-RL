import torch
import torch.optim as optim
import torch.nn.functional as F
from models.policy import GaussianPolicyNetwork
from models.value import ValueNetwork
from utils.asset import map_to_range, np_to_torch, torch_to_np, compute_GAE
from utils.buffer import RolloutBuffer
from utils.plot import plot_return
from utils.logger import TensorboardWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

class PPOAgent:
    def __init__(self, state_size, action_size, action_range, hidden_dim=[128], 
                 gamma=0.99, lr=3e-4, clip_ratio=0.2, buffer_size=2e3, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.action_range = action_range
        self.gamma = gamma
        self.lr = lr
        self.clip_ratio = clip_ratio
        self.batch_size = batch_size
        self.memory = RolloutBuffer(int(1e5))
        self.buffer_size = buffer_size

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
        states, actions, old_log_probs, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert data to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        # Obtain value estimates
        state_values = self.value_network(states)
        with torch.no_grad(): next_state_values = self.value_network(next_states)

        advantages, discounted_returns = compute_GAE(rewards, state_values, next_state_values, dones, self.gamma, lam=0.95)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        # Compute Value Loss
        value_loss = F.mse_loss(discounted_returns, state_values)

        # Update Value Network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Compute new log probs
        action_log_probs, entropy = self.new_policy.evaluate(states, actions)
        ratios = torch.exp(action_log_probs - old_log_probs)

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

        # write loss values
        self.writer.log_scalar("Loss/Policy", policy_loss, self.iter)
        self.writer.log_scalar("Loss/Entropy", entropy_loss, self.iter)
        self.writer.log_scalar("Loss/Value", value_loss, self.iter)
        self.iter += 1


    def train(self, env, episodes):
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
                action_t, action_log_prob_t = self.old_policy.select_action(state_t)
                # convert to numpy
                action = torch_to_np(action_t)
                action_log_prob = torch_to_np(action_log_prob_t)
                # map action to range
                mapped_action = map_to_range(action, self.action_range)
                # take action
                next_state, reward, done, trunc, info = env.step(mapped_action)
                # store in memory
                self.memory.push([state, action, action_log_prob, reward, next_state, done])
                state = next_state
                score += reward
                length += 1

            if len(self.memory) > self.buffer_size:
                # train agent
                for i in range(len(self.memory)//self.batch_size): self.learn()
                # clear memory
                self.memory.clear()
                # Update Old Policy
                self.old_policy.load_state_dict(self.new_policy.state_dict())

            # log episode info
            self.writer.log_scalar("Episode/Return", score, episode)
            self.writer.log_scalar("Episode/Length", length, episode)
            # store episode return
            returns.append(score)
            plot_return(returns, f'Proximal Policy Optimization (PPO) ({device})')

        env.close()
        self.writer.close()
        return returns
