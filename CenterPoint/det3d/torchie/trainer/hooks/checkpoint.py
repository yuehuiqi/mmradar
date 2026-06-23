import re
from pathlib import Path

from ..utils import master_only
from .hook import Hook


class CheckpointHook(Hook):
    def __init__(self, interval=1, save_optimizer=True, out_dir=None, max_keep_ckpts=10, **kwargs):
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max(1, int(max_keep_ckpts))
        self.args = kwargs

    @master_only
    def after_train_epoch(self, trainer):
        if not self.every_n_epochs(trainer, self.interval):
            return

        if not self.out_dir:
            self.out_dir = trainer.work_dir

        trainer.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args
        )

        checkpoint_dir = Path(self.out_dir)
        checkpoint_files = []
        for path in checkpoint_dir.glob("epoch_*.pth"):
            match = re.fullmatch(r"epoch_(\d+)\.pth", path.name)
            if match:
                checkpoint_files.append((int(match.group(1)), path))
        checkpoint_files.sort(key=lambda item: item[0])
        for _, path in checkpoint_files[:-self.max_keep_ckpts]:
            path.unlink()
