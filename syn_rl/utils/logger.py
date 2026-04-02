import os
from torch.utils.tensorboard import SummaryWriter

class TensorboardWriter:
    def __init__(self, log_dir="runs"):
        self._log_dir = log_dir
        self.run_dir = None
        self._writer = None

    def start(self, resume_dir=None):
        if resume_dir is not None:
            self.run_dir = resume_dir
        else:
            self.run_dir = self._next_run_dir(self._log_dir)
        self._writer = SummaryWriter(log_dir=self.run_dir)

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
        self._writer.add_scalar(tag, value, step)
    
    def log_histogram(self, tag, values, step):
        self._writer.add_histogram(tag, values, step)
    
    def log_image(self, tag, img_tensor, step):
        self._writer.add_image(tag, img_tensor, step)
    
    def log_text(self, tag, text_string, step):
        self._writer.add_text(tag, text_string, step)
    
    def stop(self):
        if self._writer is not None:
            self._writer.close()