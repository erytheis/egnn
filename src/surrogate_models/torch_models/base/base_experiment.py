from src.base.base_module import BaseModule
from src.utils.utils import get_abs_path


class BaseTorchExperiment(BaseModule):

    def __init__(self, model, criterion, metric_ftns, resume, device, *args, **kwargs):
        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.device = device

        if resume is not None:
            self._resume_checkpoint(resume)


    def _resume_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @checkpoint_dir.setter
    def checkpoint_dir(self, dir):
        self._checkpoint_dir = get_abs_path(dir)
