import os
from torch.utils.tensorboard import SummaryWriter

class TensorboardWriter:
    def __init__(self, log_dir="runs"):
        # Auto-increment run number: Logs/PIC-PPO -> Logs/PIC-PPO-1, PIC-PPO-2, ...
        run_dir = self._next_run_dir(log_dir)
        self.writer = SummaryWriter(log_dir=run_dir)

    def _next_run_dir(self, log_dir):
        parent = os.path.dirname(log_dir)
        base = os.path.basename(log_dir)
        if not os.path.exists(parent):
            return os.path.join(parent, f"{base}-1")
        existing = [
            d for d in os.listdir(parent)
            if os.path.isdir(os.path.join(parent, d)) and d.startswith(base)
        ]
        max_num = 0
        for d in existing:
            suffix = d[len(base):]
            if suffix.startswith("-") and suffix[1:].isdigit():
                max_num = max(max_num, int(suffix[1:]))
        return os.path.join(parent, f"{base}-{max_num + 1}")
    
    def log_scalar(self, tag, value, step):
        # Log a scalar value (e.g., loss, reward)
        self.writer.add_scalar(tag, value, step)
    
    def log_histogram(self, tag, values, step):
        # Log a histogram (e.g., weight distributions)
        self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag, img_tensor, step):
        # Log an image (e.g., state observations)
        self.writer.add_image(tag, img_tensor, step)
    
    def log_text(self, tag, text_string, step):
        # Log text information
        self.writer.add_text(tag, text_string, step)
    
    def close(self):
        self.writer.close()