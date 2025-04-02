from abc import abstractmethod

import numpy as np
import torch

from src.surrogate_models.torch_models.base.base_experiment import BaseTorchExperiment
# from src.utils.torch_utils import inf_loop
from src.surrogate_models.torch_models.logger import setup_logging
from src.utils.utils import get_abs_path


class BaseTrainer(BaseTorchExperiment):
    """
    Base class for all trainers
    """

    def __init__(self, model,
                 criterion,
                 metric_ftns,
                 optimizer,
                 writer,
                 config,
                 data_loader,
                 valid_data_loader=None,
                 lr_scheduler=None,
                 len_epoch=None,
                 clip=None,
                 device='cpu',
                 *args,
                 **kwargs):
        self.config = config
        self.clip = clip
        self.logger = config.get_logger('trainer', config['trainer']["args"]['verbosity'])
        self.optimizer = optimizer

        # logging & monitoring parameters
        self.start_epoch = 1
        self.not_improved_count = 0
        if writer is not None:
            writer.watch(model)

        super().__init__(model, criterion, metric_ftns, config.resume, device)

        self.data_loader = data_loader
        self.data_loader.dataset.clear_cache()

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            raise NotImplementedError
            # iteration-based training
            # self.data_loader = inf_loop(data_loader)
            # self.len_epoch = len_epoch
        self.lr_scheduler = lr_scheduler

        # setup validation parameters
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # setup validation and plotting periods
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer["args"]['epochs']
        self.save_period = cfg_trainer["args"]['save_period']
        self.validation_period = cfg_trainer["args"].get('validation_period', 1)

        self.plot_period = cfg_trainer["args"].get('plot_period', None)
        if self.plot_period is not None:
            assert self.plot_period % self.validation_period == 0, 'Validation period must be a multiple of plot period'

        self.monitor = cfg_trainer["args"].get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
            self.not_improved_count = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = np.inf if self.mnt_mode == 'min' else -np.inf
            self.early_stop = cfg_trainer["args"].get('early_stop', np.inf)
            if self.early_stop <= 0:
                self.early_stop = np.inf


        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = writer

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    # 
    def run(self):
        """
        Full training logic
        """

        # create checkpoint dir and dump splits
        self.config.mkdirs()
        setup_logging(self.config.log_dir)

        if hasattr(self, 'data_loader'):
            self.data_loader.dump_state(self.config.save_dir / 'loader_kwargs.json')

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off' and  epoch % self.validation_period == 0:
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    self.not_improved_count = 0
                    best = True
                    self._save_checkpoint(epoch, save_best=best, loss=log['loss'])
                else:
                    self.not_improved_count += 1

                if self.not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0 or epoch == self.epochs -1 :
                self._save_checkpoint(epoch, loss=log['loss'])

        if self.writer is not None:
            self.writer.save_table()

    def _save_checkpoint(self, epoch, loss, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config,
            'not_improved_count' : self.not_improved_count
        }

        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")
        else:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}-loss{:.4f}.pth'.format(epoch, loss))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))


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
        self.not_improved_count = checkpoint['not_improved_count'] if 'not_improved_count' in checkpoint else 0

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")

        # match the keys in state_dict and load only those that are present in the model's state_dict
        current_model_dict = self.model.state_dict()
        loaded_state_dict = checkpoint['state_dict']
        new_state_dict = {k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] for k, v in
                          zip(current_model_dict.keys(), loaded_state_dict.values())}
        self.model.load_state_dict(new_state_dict, strict=False)

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        elif self.config['optimizer'].get('reset', False):
            self.logger.info("Resetting optimizer")
        else:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.optimizer.param_groups[0]['capturable'] = True
            except ValueError:
                self.logger.warning("Warning: Optimizer parameters not being resumed.")

        # load original config to memory
        self.loaded_config = checkpoint['config']

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def to(self, device):
        self.model.to(device)


def load_model_checkpoint(model, resume_path, device):
    resume_path = get_abs_path(str(resume_path))
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    return model