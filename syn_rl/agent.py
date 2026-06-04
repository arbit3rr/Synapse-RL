from abc import ABC, abstractmethod


class RLAgent(ABC):
    """Abstract base class for Synapse-RL agents."""

    @abstractmethod
    def train(self, env, episodes: int):
        """Run the training loop for a given number of episodes."""

    @abstractmethod
    def save_checkpoint(self, filepath: str = None):
        """Persist network weights and optimizer state to disk."""

    @abstractmethod
    def load_checkpoint(self, filepath: str = None):
        """Restore network weights and optimizer state from disk."""
