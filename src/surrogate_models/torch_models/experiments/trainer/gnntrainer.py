import numpy as np
import torch

from matplotlib import pyplot as plt

from src.surrogate_models.torch_models.experiments.trainer.base_trainer import BaseTrainer
from src.surrogate_models.torch_models.visualization.plotter import WDSPlotter, plot_prediction_comparison, \
    get_output_by_index
from src.utils.torch.torch_utils import inf_loop, MetricTracker


class GNNTrainer(BaseTrainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, writer, config, device,
                 data_loader=None, valid_data_loader=None, lr_scheduler=None, len_epoch=None, clip=None, *args, **kwargs):
        super().__init__(model, criterion, metric_ftns, optimizer, writer, config, data_loader, clip=clip, *args, **kwargs )
        if writer is not None:
            writer.watch(model)

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
            *[str(loss) for loss in self.criterion],
            *[m.__name__ for m in self.metric_ftns],
            writer=self.writer)

        self.valid_metrics = MetricTracker(
            'loss',
            *['' + str(loss) for loss in self.criterion],
            *[f'{m.__name__}/combined'  # Total metrics
              for m in self.metric_ftns],
            *[f'{m.__name__}/{ds.name}'  # Filtered metrics
              for m in self.metric_ftns
              for ds in self.data_loader.datasets],
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

            with torch.no_grad():
                self.data_loader.dataset.datasets[0].pre_transform(data, inverse=True)

            loss = self.criterion(output, data)
            loss.item().backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip) if self.clip else None
            self.optimizer.step()

            output_ = output['x'] if 'x' in output else output['edge_attr']  # in case of composite output

            if batch_idx % self.log_step == 0:
                self._log_training(epoch, loss, batch_idx, data, output_)

            if batch_idx == self.len_epoch:
                break
            log = self.train_metrics.result()

        if self.do_validation and epoch % self.validation_period == 0:
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
        [self.train_metrics.update(f'{str(l)}', l.item()) for l in loss]
        y = data.y if 'node_y' not in data else data.node_y
        mask = None

        for met in self.metric_ftns:
            met_name = met.__name__
            # mask = data.mask_by_features(attr, -1) if hasattr(data, 'mask_by_features') else None
            self.train_metrics.update(met_name, met(output, y.view(output.shape), mask=mask).cpu(),
                                      n=mask.sum() if mask is not None else 1)

        self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
            epoch,
            self._progress(batch_idx),
            loss.item()))

    
    def _log_validation(self, epoch, loss, batch_idx, data, output):

        if self.writer is not None:
            self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')

        plotting_sample = data[0].clone()

        # log losses
        self.valid_metrics.update(f'loss', loss.item())
        [self.valid_metrics.update(f'{l}', l.item()) for l in loss]

        # log validation metrics
        y = data['node_y'] if 'node_y' in data else data['y']

        val_metrics = {}
        for met in self.metric_ftns:
            mask = None
            # log combined metrics
            # mask = data.mask_by_features(attr, -1) if hasattr(data, 'mask_by_features') else None
            score = met(output, y.view(output.shape), mask=mask)
            self.valid_metrics.update(f'{met.__name__}/combined', score.item(),
                                      n=mask.sum().item() if mask is not None else 1)

            # log each netwokr separately
            for ds in self.valid_data_loader.dataset.datasets:
                mask = data.mask_by_key('y', ds.name) if hasattr(data, 'mask_by_key') else None
                # mask = mask * data.mask_by_features(attr, -1) if hasattr(data, 'mask_by_features') else mask
                score = met(output, y.view(output.shape),
                            mask=mask)  # start_idx=start_idx.cpu(), end_idx=end_idx.cpu())
                score = score.cpu().numpy().item() if hasattr(score, 'numpy') else score
                if score == 0 or mask.sum() == 0:
                    continue
                self.valid_metrics.update(f'{met.__name__}/{ds.name}', score,
                                          n=mask.sum().item() if mask is not None else 1)

            self._plot(epoch, data, output)

    
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

                self.data_loader.dataset.datasets[0].pre_transform(data, inverse=True)

                loss = self.criterion(output, data)

                self._log_validation(epoch, loss, batch_idx, data, output['x'])
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


class WDSTrainer(GNNTrainer):
    """
    Trainer class for Water Distribution Networks, contain plotting logic with WNTR package
    """

    def __init__(self, show=False,save=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plotter = WDSPlotter(datasets=kwargs['data_loader'].dataset)
        self.show = show
        self.save = save

    def _plot(self, epoch, data, output, plotting_func=plot_prediction_comparison, key='x'):

        if self.plotter is not None:
            # skip plotting
            if epoch % int(self.plot_period) != 0:
                return

            # plot gradients flow
            # grad_plot = plot_grad_flow(self.model.named_parameters())

            # First plot random prediction
            tiled_predictions, tiled_true = get_output_by_index(
                data.cpu(),
                output.cpu(),
                key=key
                # size_n=2
            )
            # Plot an image for logging
            log_plot = plotting_func(tiled_predictions,
                                     tiled_true)


            # self.writer.save_table()
            if self.show:
                plt.show()
            if self.save:
                plt.savefig(f'{self.config.log_dir}/{key}_epoch_{epoch}-network_{data[0].wds_names}.png')


            # plot MAD
            network_plot = self.plotter.plot_network(np.abs(tiled_predictions - tiled_true), data[0].wds_names)

            if self.writer is not None:
                self.writer.add_row(
                    str(self.plotter),
                    epoch,
                    f"Random Sample {data[0].wds_names}",
                    self.writer.image(log_plot),
                    self.writer.image(network_plot) if network_plot is not None else None,
                    None
                )

            plt.close('all')
            return
            # # Secondly, plot the worst prediction
            # # First plot random prediction
            # tiled_predictions, tiled_true, idx = get_worst_prediction_in_batch(
            #     data.cpu(),
            #     output.cpu(),
            #     # size_n=2
            # )
            # # Plot an image for logging
            # log_plot = plotting_func(tiled_predictions,
            #                          tiled_true, batch=data, idx=idx)
            # # plot MAD
            # network_plot = self.plotter.plot_network(np.abs(tiled_predictions - tiled_true), data[idx].wds_names)
            #
            # if network_plot is None:
            #     return plt.close('all')
            #
            # if self.writer is not None:
            #     self.writer.add_row(
            #         epoch,
            #         "Worst Sample",
            #         self.writer.image(log_plot),
            #         self.writer.image(network_plot),
            #         None
            #     )


class ANNTrainer(GNNTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plotter = WDSPlotter(datasets=kwargs['data_loader'].dataset)

    def _plot(self, epoch, data, output, plotting_func=plot_prediction_comparison):
        if self.plotter is not None:
            # skip plotting
            if epoch % int(self.plot_period) != 0:
                return

            # First lot random prediction
            tiled_predictions, tiled_true = output, data.y
            # plot MAD
            network_plot = self.plotter.plot_network(np.abs(tiled_predictions[0] - tiled_true[0]), data[0].wds_names[0])
            # Plot an image for logging
            log_plot = plotting_func(tiled_predictions,
                                     tiled_true,
                                     batch=data, idx=0)
            self.writer.add_row(
                epoch,
                "Random Sample",
                self.writer.image(log_plot),
                self.writer.image(network_plot),
                None
            )

            plt.close('all')
