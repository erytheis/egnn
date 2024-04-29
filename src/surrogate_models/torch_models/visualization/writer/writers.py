import importlib
from datetime import datetime

import wandb


class BaseWriter:

    def __init__(self, config, logger, enabled):
        self.config = config.config
        self.logger = logger
        if enabled:
            self.log_dir = str(config.log_dir)

        self.step = 0
        self.mode = ''

        self.timer = datetime.now()

    def watch(self, *args):
        pass

    def log(self, *args):
        pass


class WandBWriter(BaseWriter):

    def __init__(self, config, logger, enabled, init=True, agent=False, layer_inspection_functions=None):
        super().__init__(config, logger, enabled)

        self.writer_ftns = {"add_row", "add_histogram"}
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.layer_inspection_functions = layer_inspection_functions

        if not enabled: return

        # Retrieve vizualization writer.
        succeeded = False
        try:
            if init:
                wandb.init(
                    project=config['name'],
                    config=self.config if not agent else None,
                    dir=config.base_dir,
                    settings=wandb.Settings(start_method='fork'))
            self.writer = wandb
            succeeded = True
        except ImportError:
            succeeded = False

        if not succeeded:
            message = "Warning: visualization (WandB) is configured to use, but currently not installed on " \
                      "this machine. Please install wandb"
            self.logger.warning(message)

        self.init_tables()

    def image(self, image):
        return wandb.Image(image)

    def init_tables(self):
        self.tables = {}
        self.tables['wds_plots'] = wandb.Table(
            columns=["epoch", "Sample", "first_prediction", "error on nodes", "true_values"])
        if self.layer_inspection_functions is not None:
            self.tables['layer_inspection'] = wandb.Table(
                columns=["epoch", "Sample"] + self.layer_inspection_functions)

    def save_table(self):
        [self.writer.log({k: t}) for k, t in self.tables.items()]

    def log(self, item, validation=False):
        self.writer.log(item)

    # def add_prediction_comparison(self, epoch, batch_id, prediction, true_labels):
    #     self.tables.add_data(epoch, batch_id, prediction, true_labels)

    def set_step(self, step, mode='train', *args, **kwargs):
        pass

    def add_row(self, plotter_name, *row, **kwargs):

        """Add a row to a table and save it. A workaround with adding new table everytime
        because wandb does not support adding rows to a table live

        """
        self.tables[plotter_name] = wandb.Table(
            columns=self.tables[plotter_name].columns,
            data=self.tables[plotter_name].data
        )
        self.tables[plotter_name].add_data(*row, **kwargs)
        self.writer.log({plotter_name: self.tables[plotter_name]})


    def histogram(self, data):
        return wandb.Histogram(data)

    def watch(self, *args, **kwargs):
        log = 'all'
        wandb.watch(*args, **kwargs, log=log)


class TensorboardWriter(BaseWriter):
    def __init__(self, log_dir, logger, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.torch_utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                          "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                          "version >= 1.1 to use 'torch.torch_utils.tensorboard' or turn off the option in the 'config.json' file."
                logger.warning(message)

        self.step = 0
        self.mode = ''

        self.writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()
