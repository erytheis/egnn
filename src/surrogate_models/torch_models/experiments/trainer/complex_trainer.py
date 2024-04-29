import numpy as np
import torch
from matplotlib import pyplot as plt

from src.surrogate_models.torch_models.experiments.trainer.base_trainer import BaseTrainer
from src.surrogate_models.torch_models.visualization.plotter import WDSPlotter, plot_prediction_comparison, \
    get_output_by_index, get_worst_prediction_in_batch
from src.utils.torch.torch_utils import inf_loop, MetricTracker


class ComplexTrainer(BaseTrainer):
    cell_names = {0: 'node',
                  1: 'edge',
                  2: 'cells'}

    def __init__(self, model, criterion, metric_ftns, optimizer, writer, config, device,
                 data_loader, valid_data_loader=None, y_dim=1, lr_scheduler=None, len_epoch=None, clip=None, *args,
                 **kwargs):
        super().__init__(model, criterion, metric_ftns, optimizer, writer, config, clip=None, *args, **kwargs)
        if writer is not None:
            writer.watch(model)

        self.y_dim = y_dim
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker(
            'loss',
            *[f'{self.cell_names[dim]}/{str(loss)}' for loss in self.criterion for dim in range(self.y_dim)],
            *[f'{self.cell_names[dim]}/{m.__name__}' for m in self.metric_ftns for dim in range(self.y_dim)],
            writer=self.writer)

        self.valid_metrics = MetricTracker(
            'loss',
            *[f'{self.cell_names[dim]}/{str(loss)}' for loss in self.criterion for dim in range(self.y_dim)],
            *[f'{self.cell_names[dim]}/{m.__name__}/combined'  # Total metrics
              for m in self.metric_ftns
              for dim in range(self.y_dim)],
            *[f'{self.cell_names[dim]}/{m.__name__}/{ds.name}'  # Filtered metrics
              for m in self.metric_ftns
              for ds in self.data_loader.datasets
              for dim in range(self.y_dim)],
            writer=self.writer,
            validation=True)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader):
            data = data.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, data)
            loss.item().backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip) if self.clip else None
            self.optimizer.step()

            if batch_idx % self.log_step == 0:
                self._log_training(epoch, loss, batch_idx, data, output)

            if batch_idx == self.len_epoch:
                break
            log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _log_training(self, epoch, loss, batch_idx, data, output):
        if self.writer is not None:
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

        # log losses
        self.train_metrics.update('loss', loss.item())
        [self.train_metrics.update(f'{l}', l.item()) for l in loss]

        for dim, prediction in enumerate(output):
            if data.cochains[dim]['y'] is not None:
                y = data.cochains[dim]['y']
            else:
                continue

            for met in self.metric_ftns:
                met_name = f'{self.cell_names[dim]}/{met.__name__}'
                self.train_metrics.update(met_name, met(prediction.cpu(), y.view(prediction.shape).cpu()))

        self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
            epoch,
            self._progress(batch_idx),
            loss.item()))

    def _log_validation(self, epoch, loss, batch_idx, data, output):
        # self.writer.add_image('input', make_grid(data.x.cpu(), nrow=1, normalize=True))
        if self.writer is not None:
            self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')

        # log losses
        self.valid_metrics.update(f'loss', loss.item())
        [self.valid_metrics.update(f'{l}', l.item()) for l in loss]

        for dim, prediction in enumerate(output):
            if data.cochains[dim]['y'] is not None:
                y = data.cochains[dim]['y']
            else:
                continue

            prediction = output[dim].cpu()

            # log validation metrics
            val_metrics = {}
            for met in self.metric_ftns:
                met_name = f'{self.cell_names[dim]}/{met.__name__}'
                val_metrics[f'{met_name}/combined'] = met(prediction, y.view(prediction.shape).cpu())

                for ds in self.data_loader.datasets:
                    mask = data.mask_by_key('y', ds.name, dim).squeeze().cpu() if hasattr(data, 'mask_by_key') else None
                    val_metrics[f'{met_name}/{ds.name}'] = met(prediction, y.view(prediction.shape).cpu(), mask=mask)
            self.valid_metrics.update_metrics(val_metrics)

        self._plot(epoch, data, output[0])

    def _plot(self):
        pass

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                data = data.to(self.device)
                output = self.model(data)

                loss = self.criterion(output, data)
                self._log_validation(epoch, loss, batch_idx, data, output)
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class WDSComplexTrainer(ComplexTrainer):
    """
    Trainer class for Water Distribution Networks, contain plotting logic with WNTR package
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plotter = WDSPlotter(datasets=kwargs['data_loader'].dataset)

    def _plot(self, epoch, data, output, plotting_func=plot_prediction_comparison):
        if self.plotter is not None:
            #skip plotting
            if epoch % int(self.save_period / 10) != 0:
                return
            i = 0
            # First get labesl and predictions
            node_prediction, node_label = get_output_by_index(
                data.cochains[0],
                output.cpu(),
                y_size=data.cochains[0].ptr[1]
                # size_n=2
            )

            if self.y_dim > 1:
                # Get labels for edge prediction
                edge_prediction, edge_label = get_output_by_index(
                    data.cochains[1],
                    output.cpu(),
                    y_size=data.cochains[1].ptr[1]
                    # size_n=2
                )
                edge_index = edge_index = data.cochains[1].boundary_index[:, :data.cochains[1].ptr[1] * 2]
                edge_index = torch.stack([edge_index[0, ::2], edge_index[0, 1::2]])
                edge_attributes = np.abs(edge_prediction - edge_label)
            else:
                edge_prediction, edge_label, edge_index = None, None, None
                edge_attributes = None

            # Plot errors
            error_plot = plotting_func(node_prediction,
                                       node_label,
                                       batch=data.cochains[0], idx=0)
            error_network_plot = self.plotter.plot_network(np.abs(node_prediction - node_label),
                                                           data.wds_names[0],
                                                           edge_attributes=edge_attributes,
                                                           edge_index=edge_index
                                                           )
            # plot MAD
            if error_network_plot is not None:
                error_network_plot.set_title(f'Absolute error {data.wds_names[0]}')
            else:
                return plt.close('all')

            # Plot absolute values
            true_labels_network_plot = self.plotter.plot_network(node_label,
                                                                 data.wds_names[0],
                                                                 edge_label,
                                                                 edge_index=edge_index

                                                                 )
            # plot MAD
            true_labels_network_plot.set_title(f'True values {data.wds_names[0]}')

            if self.writer is not None:
                self.writer.add_row(
                    epoch,
                    "Random Sample",
                    self.writer.image(error_plot),
                    self.writer.image(error_network_plot),
                    self.writer.image(true_labels_network_plot)
                )

            plt.close('all')
