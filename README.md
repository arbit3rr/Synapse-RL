<div align="center">
  <a href="https://github.com/amirhosseinh77/Synapse-RL">
    <img width="100%" src="https://user-images.githubusercontent.com/56114938/235786557-186fe616-f0ab-4a14-95d5-c95817062942.png">
  </a>
  <h1>Synapse-RL</h1>
  <p>A clean, modular PyTorch library for deep reinforcement learning</p>

  <a href="https://colab.research.google.com/github/amirhosseinh77/Synapse-RL/blob/main/SYNAPES_tutorial.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
  <a href="https://doi.org/10.5281/zenodo.8010048">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.8010048.svg" alt="DOI">
  </a>
</div>

---

**Synapse-RL** is a lightweight, research-friendly PyTorch library for deep reinforcement learning. It provides clean implementations of foundational and modern RL algorithms with a consistent interface, TensorBoard logging, and [Gymnasium](https://gymnasium.farama.org/) compatibility.

## Features

- **8 algorithms** covering both discrete and continuous action spaces
- **Consistent API** — every algorithm exposes the same `train / save_checkpoint / load_checkpoint` interface
- **TensorBoard integration** — automatic per-run logging with auto-incrementing directories
- **Checkpoint system** — save and resume training at any point
- **Gymnasium compatible** — works with any `gym` / `gymnasium` environment
- **GPU support** — automatic CUDA detection and device placement

## Installation

```bash
git clone https://github.com/amirhosseinh77/Synapse-RL.git
cd Synapse-RL
pip install -r requirements.txt
```

Or install as an editable package so you can import it from anywhere:

```bash
pip install -e .
```

## Quick Start

```python
import gymnasium as gym
from syn_rl import SAC

env = gym.make("Pendulum-v1")
state_size  = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

agent = SAC(
    state_size, action_size,
    action_range=[env.action_space.low, env.action_space.high],
    hidden_dim=[256, 256],
)

returns = agent.train(env, episodes=500)
```

## Algorithms

| Algorithm | Import | Action Space | Reference |
|---|---|---|---|
| Deep Q-Network | `DQN` | Discrete | [Mnih et al., 2015](https://www.nature.com/articles/nature14236) |
| Policy Gradient (REINFORCE) | `PolicyGradient` | Discrete | [Williams, 1992](https://link.springer.com/article/10.1007/BF00992696) |
| Advantage Actor-Critic | `ActorCritic` | Discrete | [Mnih et al., 2016](https://arxiv.org/abs/1602.01783) |
| Deep Deterministic Policy Gradient | `DDPG` | Continuous | [Lillicrap et al., 2015](https://arxiv.org/abs/1509.02971) |
| Soft Actor-Critic | `SAC` | Continuous | [Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290) |
| SAC with Value Network | `SAC_VALUE` | Continuous | [Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290) |
| Proximal Policy Optimization (step) | `PPO` | Continuous | [Schulman et al., 2017](https://arxiv.org/abs/1707.06347) |
| Proximal Policy Optimization (episode) | `PPO_EP` | Continuous | [Schulman et al., 2017](https://arxiv.org/abs/1707.06347) |

## API Reference

### Common Interface

Every algorithm shares the same three-method interface:

```python
returns = agent.train(env, episodes=1000)   # run training loop; returns list of episode returns
agent.save_checkpoint(filepath)             # save networks + optimizers to .pth file
agent.load_checkpoint(filepath)             # restore from .pth file and resume logging
```

`save_checkpoint` / `load_checkpoint` default to `<log_dir>/checkpoint.pth` when `filepath` is omitted.

---

### Discrete Algorithms

#### DQN

```python
from syn_rl import DQN

agent = DQN(
    state_size,
    action_size,
    hidden_dim=[128],        # list of hidden layer widths
    gamma=0.99,              # discount factor
    epsilon=1.0,             # initial ε-greedy exploration rate
    epsilon_min=0.05,        # floor on epsilon
    epsilon_decay=0.995,     # multiplicative decay applied each episode
    lr=3e-4,                 # Adam learning rate
    tau=0.005,               # soft-update coefficient for target network
    buffer_size=1e5,         # replay buffer capacity
    batch_size=256,
)
```

#### PolicyGradient

```python
from syn_rl import PolicyGradient

agent = PolicyGradient(
    state_size,
    action_size,
    hidden_dim=[128],
    gamma=0.99,
    lr=1e-3,
)
```

#### ActorCritic (A2C)

```python
from syn_rl import ActorCritic

agent = ActorCritic(
    state_size,
    action_size,
    hidden_dim=[128],
    gamma=0.99,
    lr=1e-3,
)
```

---

### Continuous Algorithms

All continuous algorithms need `action_range` — the low/high bounds of the environment's action space:

```python
action_range = [env.action_space.low, env.action_space.high]
```

Actions are internally mapped from the network's `[-1, 1]` tanh output to this range.

#### DDPG

```python
from syn_rl import DDPG

agent = DDPG(
    state_size, action_size, action_range,
    hidden_dim=[128],
    gamma=0.99,
    min_uncertainty=0.1,       # minimum Gaussian exploration noise
    uncertainty_decay=0.998,   # multiplicative noise decay per episode
    lr=3e-4,
    tau=0.005,
    buffer_size=1e5,
    batch_size=256,
)
```

#### SAC

```python
from syn_rl import SAC

agent = SAC(
    state_size, action_size, action_range,
    hidden_dim=[128],
    alpha=0.1,       # initial entropy temperature (auto-tuned during training)
    gamma=0.99,
    lr=3e-4,
    tau=0.005,
    buffer_size=1e5,
    batch_size=256,
)
```

#### SAC_VALUE

A variant of SAC that uses an explicit state-value network (V) instead of a second Q-network as the bootstrap target. The entropy temperature `alpha` is fixed rather than learned.

```python
from syn_rl import SAC_VALUE

agent = SAC_VALUE(
    state_size, action_size, action_range,
    hidden_dim=[128],
    alpha=0.1,
    gamma=0.99,
    lr=3e-4,
    tau=0.005,
    buffer_size=1e5,
    batch_size=256,
)
```

#### PPO (step-based)

Updates the policy on a rolling replay buffer every step. Supports both clipped-surrogate (`'clip'`) and KL-penalty (`'penalty'`) objectives.

```python
from syn_rl import PPO

agent = PPO(
    state_size, action_size, action_range,
    hidden_dim=[128],
    gamma=0.99,
    lam=0.95,               # GAE lambda
    lr=3e-4,
    policy_update_freq=100, # steps between old-policy sync
    buffer_size=2000,
    batch_size=256,
    alg='clip',             # 'clip' or 'penalty'
    clip_ratio=0.1,
    beta=1.0,               # initial KL penalty coefficient (penalty mode only)
    target_kl=0.01,         # adaptive beta target (penalty mode only)
)
```

#### PPO_EP (episode-based)

Collects a full buffer of experience then runs K gradient epochs before clearing and syncing the old policy.

```python
from syn_rl import PPO_EP

agent = PPO_EP(
    state_size, action_size, action_range,
    hidden_dim=[128],
    gamma=0.99,
    lam=0.95,
    lr=3e-4,
    clip_ratio=0.1,
    K_epochs=100,     # gradient update passes per buffer
    buffer_size=2000,
)
```

---

## Checkpointing

```python
# Save after training
agent.train(env, episodes=300)
agent.save_checkpoint("models/sac_pendulum.pth")

# Resume later
agent.load_checkpoint("models/sac_pendulum.pth")
agent.train(env, episodes=200)   # continues from where it left off
```

DDPG, SAC, and PPO also auto-save a `best_model.pth` inside the log directory whenever a periodic evaluation episode achieves a new best return.

---

## TensorBoard

All runs are logged automatically. Start TensorBoard with:

```bash
tensorboard --logdir Logs/
```

Logs are written to `Logs/<Algorithm>-<run>/` and include:

| Tag | Algorithms |
|---|---|
| `Episode/Return` | all |
| `Episode/Length` | all |
| `Episode/Return Eval` | DDPG, SAC, PPO |
| `Episode/Epsilon` | DQN |
| `Loss/Policy` or `Loss/Actor` | all |
| `Loss/Critic` / `Loss/Value` / `Loss/Q1` / `Loss/Q2` | actor-critic algorithms |
| `Entropy/Alpha`, `Entropy/Alpha_Loss` | SAC |
| `KL/Approx`, `Beta` | PPO (penalty mode) |

---

## Project Structure

```
syn_rl/
├── __init__.py          # public API — imports all algorithm classes
├── agent.py             # RLAgent abstract base class
├── algorithm/
│   ├── dqn.py           # Deep Q-Network
│   ├── pg.py            # Policy Gradient (REINFORCE)
│   ├── a2c.py           # Advantage Actor-Critic
│   ├── ddpg.py          # Deep Deterministic Policy Gradient
│   ├── sac.py           # Soft Actor-Critic (auto-alpha)
│   ├── sac_v.py         # SAC with explicit value network
│   ├── ppo.py           # PPO – step-based buffer
│   └── ppo_ep.py        # PPO – episode-based buffer with K epochs
├── network/
│   ├── policy.py        # CategoricalPolicyNetwork · DeterministicPolicyNetwork · GaussianPolicyNetwork
│   └── value.py         # ValueNetwork · QNetwork · DQNetwork
└── utils/
    ├── asset.py         # tensor helpers, compute_rewards_to_go, compute_GAE
    ├── buffer.py        # ExpBuffer — circular replay buffer
    ├── logger.py        # TensorboardWriter — auto-incrementing run directories
    └── plot.py          # plot_return — real-time episode return plot
```

---

## Citation

If you use Synapse-RL in your research, please cite:

```bibtex
@software{heydarian_ardakani_synapse_rl,
  author = {Heydarian Ardakani, Amirhossein},
  title  = {{Synapse RL}: A PyTorch Framework for Reinforcement Learning},
  doi    = {10.5281/zenodo.8010048},
  url    = {https://github.com/amirhosseinh77/Synapse-RL},
}
```
