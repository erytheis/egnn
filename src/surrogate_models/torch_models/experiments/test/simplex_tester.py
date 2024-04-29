from typing import Optional, List

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.surrogate_models.torch_models.experiments import SimplexTrainer, WDSSimplexTrainer
from src.surrogate_models.torch_models.experiments.test import TorchTester
from src.surrogate_models.torch_models.logger import setup_logging
from src.surrogate_models.torch_models.model.simplicial.pressure_reconstruction import from_flowrates_to_heads
from src.surrogate_models.torch_models.visualization.plotter import WDSPlotter, plot_prediction_comparison, \
    get_output_by_index
from src.utils.torch_utils import MetricTracker
from src.utils.utils import get_abs_path


class WDSSimplexTester(TorchTester):
    simplex_names = {0: 'node',
                     1: 'edge',
                     'x': 'node',
                     'edge_attr': 'edge'}
    # probably better to put it in a separate function
    mask_idx = {'x': 2,
                'edge_attr': 2}
    mask_value = {'x': 1,
                  'edge_attr': 0}

    def __init__(self,
                 y_dim: Optional[List[int]] = None,
                 show=False,
                 save=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.plotter = WDSPlotter(datasets=kwargs['data_loader'].dataset)
        self.show = show
        self.save = save

        self.y_dim = [1] if y_dim is None else y_dim

        self.metrics = MetricTracker(
            'loss',
            *[f'{str(loss)}' for loss in self.criterion],
            *[f'{self.simplex_names[dim]}/{m.__name__}/combined'  # Total metrics
              for m in self.metric_ftns
              for dim in self.y_dim],
            *[f'{self.simplex_names[dim]}/{m.__name__}/{ds.name}'  # Filtered metrics
              for m in self.metric_ftns
              for ds in self.data_loader.dataset.datasets
              for dim in self.y_dim])

    def run(self):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation

        # create checkpoint dir and dump splits
        """
        self.config.mkdirs()
        setup_logging(self.config.log_dir)

        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():
            for batch_idx, data in enumerate(self.data_loader):
                data = data.to(self.device)
                output = self.model(data)

                # temp solution
                transforms = self.data_loader.dataset.datasets[0].pre_transform
                transforms(data, inverse=True)
                loss = self.criterion(output, data)
                output = from_flowrates_to_heads(data, output)

                self._log_validation(loss, batch_idx, data, output)

        log = self.metrics.result()
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

    def _log_validation(self, loss, batch_idx, data, output):
        """
        Log validation metrics

        :param epoch: Integer, current training epoch.
        :param loss: Loss value.
        :param batch_idx: Integer, current batch index.
        :param data: Input data.
        :param output: Output data.
        """

        # log losses
        self.metrics.update(f'loss', loss.item())
        [self.metrics.update(f'{l}', l.item()) for l in loss]

        # log validation metrics
        for i, (attr, prediction) in enumerate(output.items()):
            attr_name = f'{self.simplex_names[attr]}_y'

            y = data[attr_name]

            if y is None:
                continue

            # log combined metrics
            val_metrics = {}
            for met in self.metric_ftns:
                met_name = f'{self.simplex_names[attr]}/{met.__name__}'
                val_metrics[f'{met_name}/combined'] = met(prediction.cpu(), y.view(prediction.shape).cpu())

                # log each network separately
                for ds in self.data_loader.dataset.datasets:
                    mask = data.mask_by_key(attr_name, ds.name).cpu() if hasattr(data, 'mask_by_key') else None

                    if hasattr(data, 'mask_by_features'):
                        mask = mask * data.mask_by_features(attr, self.mask_idx[attr], self.mask_value[attr]).cpu()

                    score = met(prediction.cpu(), y.view(prediction.shape).cpu(),
                                mask=mask)  # start_idx=start_idx.cpu(), end_idx=end_idx.cpu())
                    score = score.numpy().item() if hasattr(score, 'numpy') else score
                    if score == 0:
                        continue
                    self.metrics.update(f'{met_name}/{ds.name}', score,
                                        n=mask.sum().item() if mask is not None else 1)

        # plot results
        [self._plot(0, data, output[key].cpu(), key=key) for key in output.keys()]

    def _plot(self, epoch, data, output, plotting_func=plot_prediction_comparison, key='x'):

        if self.plotter is not None and (self.show or self.save):
            # get number of virtual nodes

            try:
                # plot gradients flow

                # First plot random prediction
                tiled_predictions, tiled_true = get_output_by_index(
                    data.cpu(),
                    output.cpu(),
                    key=key,
                    label_key=f'{self.simplex_names[key]}_y',
                    # size_n=2
                )
                # Plot an image for logging
                log_plot = plotting_func(tiled_predictions,
                                         tiled_true,
                                         )
                # plot MAD
                network_plot = self.plotter.plot_network(np.abs(tiled_predictions - tiled_true), data[0].wds_names,
                                                         )

                if hasattr(self, 'writer') and self.writer is not None:
                    self.writer.add_row(
                        str(self.plotter),
                        epoch,
                        f"Random Sample {data[0].wds_names}",
                        self.writer.image(log_plot),
                        self.writer.image(network_plot) if network_plot is not None else None,
                        None
                    )
                    # self.writer.save_table()

                if self.show:
                    plt.show()
                if self.save:
                    plt.savefig(f'{self.config.log_dir}/{key}_epoch_{epoch}-network_{data[0].wds_names}.png')
                plt.close('all')
            except ValueError as e:
                print(e)
            return


def validate_sign_equivariance(batch, model, idx_to_flip=0):
    batch_old = batch.clone()

    # flip all edges and edge_attr with a given idx_to_flip
    batch.edge_attr[:, idx_to_flip] *= -1
    batch.edge_index[0, :] = batch.edge_index[1, :]

    batch.edge_index[:, batch.to_flip] = torch.flip(batch.edge_index[:, batch.to_flip], [0])

    for key, value in batch.items():
        if 'weight' in key:
            batch[key] = batch[key] * -1
