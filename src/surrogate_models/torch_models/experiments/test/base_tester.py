from tqdm import tqdm
import torch

from src.surrogate_models.torch_models.base.base_experiment import BaseTorchExperiment
from src.utils.utils import get_abs_path


class TorchTester(BaseTorchExperiment):

    def __init__(self, model, criterion, metric_ftns, config, device,
                 data_loader, logger = None, *args, **kwargs):
        self.config = config
        self.logger = logger or config.get_logger('test', config['experiments']['verbosity'])

        super().__init__(model, criterion, metric_ftns, config.resume, device)

        self.data_loader = data_loader
        self.device = device
        self.checkpoint_dir = config.save_dir

    def run(self):
        total_loss = 0.0
        total_metrics = torch.zeros(len(self.metric_ftns))

        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(self.data_loader)):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # save sample images, or do something with output here

                # computing loss, metrics on test set
                loss = self.criterion(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(self.metric_ftns):
                    total_metrics[i] += metric(output, target) * batch_size

        n_samples = len(self.data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(self.metric_ftns)
        })
        self.logger.info(log)

    def _resume_checkpoint(self, resume_path):

        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = get_abs_path(str(resume_path))

        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.config['device'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")

        # match the keys in state_dict and load only those that are present in the model's state_dict
        loaded_state_dict = checkpoint['state_dict']

        self.model.load_state_dict(loaded_state_dict, strict=True)

        # load original config to memory
        self.loaded_config = checkpoint['config']

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class GNNTester(TorchTester):

    def run(self):
        total_loss = 0.0
        total_metrics = torch.zeros(len(self.metric_ftns))

        with torch.no_grad():
            for batch_idx, data in enumerate(self.data_loader):
                data = data.to(self.device)
                output = self.model(data.x, data.edge_index, data.edge_attr)
                #
                # save sample images, or do something with output here
                #
                # computing loss, metrics on test set
                loss = self.criterion(output, data.y.unsqueeze(-1))
                batch_size = len(data)
                total_loss += loss.item() * batch_size

                for i, metric in enumerate(self.metric_ftns):
                    total_metrics[i] += metric(output, data.y.unsqueeze(-1))

        print('batch size:',batch_size)
        print('output size:',data.y.unsqueeze(-1).shape)
        print('labels size:',output.shape)
        n_samples = len(self.data_loader)
        print('num samples:', n_samples)
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(self.metric_ftns)
        })
        self.logger.info(log)

