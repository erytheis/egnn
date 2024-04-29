import os

import numpy as np
import torch
from line_profiler_pycharm import profile
from matplotlib import pyplot as plt
from torch.profiler import tensorboard_trace_handler

import src.surrogate_models.torch_models.visualization.plotter
from src.surrogate_models.torch_models.dataset.transforms import VirtualNode
from src.surrogate_models.torch_models.experiments.trainer.base_trainer import BaseTrainer
from src.surrogate_models.torch_models.model.simplicial.simplicial_conv import attr_to_dim
from src.surrogate_models.torch_models.model.simplicial.pressure_reconstruction import iterative_shortest_path_distance, \
    from_flowrates_to_heads
from src.surrogate_models.torch_models.visualization.plotter import WDSPlotter, plot_prediction_comparison
from src.utils.torch.torch_utils import MetricTracker


class SimplexTrainer(BaseTrainer):
    simplex_names = {0: 'node',
                     1: 'edge',
                     'x': 'node',
                     'edge_attr': 'edge'}
    mask_idx = {'x': 2,
                'edge_attr': 2}
    mask_value = {'x': 1,
                  'edge_attr': 0}

    def __init__(self,
                 y_dim=None, *args,
                 inverse_transform=False,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.y_dim = [1] if y_dim is None else y_dim
        self.inverse_transform = inverse_transform

        self.train_metrics = MetricTracker(
            'loss',
            *[f'{str(loss)}' for loss in self.criterion],
            *[f'{self.simplex_names[dim]}/{m.__name__}' for m in self.metric_ftns for dim in self.y_dim],
            writer=self.writer)

        self.valid_metrics = MetricTracker(
            'loss',
            *[f'{str(loss)}' for loss in self.criterion],
            *[f'{self.simplex_names[dim]}/{m.__name__}/combined'  # Total metrics
              for m in self.metric_ftns
              for dim in self.y_dim],
            *[f'{self.simplex_names[dim]}/{m.__name__}/{ds.name}'  # Filtered metrics
              for m in self.metric_ftns
              for ds in self.data_loader.datasets
              for dim in self.y_dim],
            writer=self.writer,
            validation=True)

    @profile
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader):

            # prepare
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()

            data = data.to(self.device)

            output = self.model(data)



            loss = self.criterion(output, data)  # * data.num_graphs # Multiply by batch size

            loss.item().backward()

            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip) if self.clip else None
            self.optimizer.step()

            if batch_idx % self.log_step == 0:
                with torch.no_grad():
                    self._log_training(epoch, loss, batch_idx, data, output)

            if batch_idx == self.len_epoch:
                break

        # log trainig
        log = self.train_metrics.result()

        if self.do_validation and epoch % self.validation_period == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.logger.debug(f'Learning rate: {self.lr_scheduler.get_last_lr()}')
        return log

    @profile
    def _log_training(self, epoch, loss, batch_idx, data, output):
        if self.writer is not None:
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

        # log losses
        self.train_metrics.update('loss', loss.item())
        [self.train_metrics.update(f'{str(l)}', l.item()) for l in loss]
        for i, (attr, prediction) in enumerate(output.items()):

            if 'intermediate' in attr:
                continue

            y = data[f'{self.simplex_names[attr]}_y']
            mask = None
            if y is None:
                continue

            for met in self.metric_ftns:
                if attr == 'x':
                    mask_idx, mask_value = 2, 1
                else:
                    mask_idx, mask_value = -1, 0

                met_name = f'{self.simplex_names[attr]}/{met.__name__}'
                mask = data.mask_by_features(attr, column_idx=mask_idx, value=mask_value) if hasattr(data, 'mask_by_features') else None
                score = met(prediction, y.view(prediction.shape), mask=mask)

                self.train_metrics.update(met_name, score.cpu(),
                                          n=mask.sum() if mask is not None else 1)

        self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
            epoch,
            self._progress(batch_idx),
            loss.item()))

    @profile
    def _log_validation(self, epoch, loss, batch_idx, data, output):

        if self.writer is not None:
            self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')

        plotting_sample = data[0].clone()

        # log losses
        self.valid_metrics.update(f'loss', loss.item())
        [self.valid_metrics.update(f'{l}', l.item()) for l in loss]

        # log validation metrics
        for i, (attr, prediction) in enumerate(output.items()):

            if 'intermediate' in attr:
                continue

            attr_name = f'{self.simplex_names[attr]}_y'

            y = data[attr_name]

            if y is None:
                continue

            val_metrics = {}
            for met in self.metric_ftns:
                mask = None
                # log combined metrics
                met_name = f'{self.simplex_names[attr]}/{met.__name__}'
                mask = data.mask_by_features(attr, -1) if hasattr(data, 'mask_by_features') else None

                score = met(prediction, y.view(prediction.shape), mask=mask)

                self.valid_metrics.update(f'{met_name}/combined', score.item(),
                                          n=mask.sum().item() if mask is not None else 1)

                # log each netwokr separately
                for ds in self.valid_data_loader.dataset.datasets:
                    mask = data.mask_by_key(attr_name, ds.name) if hasattr(data, 'mask_by_key') else None
                    mask = mask * data.mask_by_features(attr, -1) if hasattr(data, 'mask_by_features') else mask
                    score = met(prediction, y.view(prediction.shape),
                                mask=mask)  # start_idx=start_idx.cpu(), end_idx=end_idx.cpu())
                    score = score.cpu().numpy().item() if hasattr(score, 'numpy') else score
                    if score == 0 or mask.sum() == 0:
                        continue
                    self.valid_metrics.update(f'{met_name}/{ds.name}', score,
                                              n=mask.sum().item() if mask is not None else 1)


    @profile
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
                self._plot(epoch, data[0])
                data = data.to(self.device)

                # validate
                output = self.model(data)

                loss = self.criterion(output, data)

                self._log_validation(epoch, loss, batch_idx, data, output)
                # plot results

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

    def _plot(self, *args, **kwargs):
        pass

    def _save_checkpoint(self, *args, **kwargs):
        super()._save_checkpoint(*args, **kwargs)
        transform = self.data_loader.dataset.transform
        pre_transform = self.data_loader.dataset.pre_transform

        filename = str(self.checkpoint_dir / 'transform.pth')
        if not os.path.exists(filename):
            torch.save(transform, filename)

        # save if doesn't exist
        filename = str(self.checkpoint_dir / 'pre_transform.pth')
        if not os.path.exists(filename):
            torch.save(pre_transform, filename)


class WDSSimplexTrainer(SimplexTrainer):
    """
    Trainer class for Water Distribution Networks, contain plotting logic with WNTR package
    """

    def __init__(self, show=False, save=False,
                 cache_clear_period=None,  # clears transformed cache to update it with new data
                 reconstruct_pressures=False,
                 inverse_transform_train=False,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.cache_clear_period = cache_clear_period
        self.reconstruct_pressures = reconstruct_pressures
        self.inverse_transform_train= inverse_transform_train
        self.dataset = kwargs['data_loader'].dataset
        self.plotter = WDSPlotter(datasets=kwargs['data_loader'].dataset)
        self.show = show
        self.save = save

        # check if has virtual nodes
        self.virtual_idx = None
        for t in self.dataset.datasets[0].transform:
            if isinstance(t, VirtualNode):
                self.virtual_idx = [t.virtual_idx][0]

    def _train_epoch(self, epoch):
        if self.cache_clear_period is not None and epoch % self.cache_clear_period == 0:
            self.data_loader.dataset.clear_cache()

        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader):

            # prepare
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()

            data = data.to(self.device)

            output = self.model(data)

            if self.inverse_transform_train:
                with torch.no_grad():
                    self.data_loader.dataset.datasets[0].pre_transform(data, inverse=True)

            loss = self.criterion(output, data)  # * data.num_graphs # Multiply by batch size
            loss.item().backward()

            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip) if self.clip else None
            self.optimizer.step()

            if batch_idx % self.log_step == 0:
                with torch.no_grad():
                    self._log_training(epoch, loss, batch_idx, data, output)

            if batch_idx == self.len_epoch:
                break

        # log trainig
        log = self.train_metrics.result()

        if self.do_validation and epoch % self.validation_period == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.logger.debug(f'Learning rate: {self.lr_scheduler.get_last_lr()}')
        return log

    @profile
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
                self._plot(epoch, data[0])
                data = data.to(self.device)

                # validate
                output = self.model(data)

                if self.inverse_transform:
                    with torch.no_grad():
                        self.data_loader.dataset.datasets[0].pre_transform(data, inverse=True)
                if self.reconstruct_pressures:
                    output = from_flowrates_to_heads(data, output)
                        # output = iterative_shortest_path_distance(data, output, num_iterations=self.model.num_layers,
                        #                                           aggr=self.model.aggregator)

                loss = self.criterion(output, data)

                self._log_validation(epoch, loss, batch_idx, data, output)
                # plot results

        return self.valid_metrics.result()

    def _plot(self, epoch, data, plotting_func=plot_prediction_comparison):

        if self.plotter is not None and (self.show or self.save):
            output_ = self.model(data)
            # skip plotting
            if epoch % int(self.plot_period) != 0:
                return

            # get number of virtual nodes
            virtual = {}
            if self.virtual_idx is not None:
                virtual = {key: data[key][:, self.virtual_idx[key]].cpu().numpy() for key in self.virtual_idx.keys()}

            for key in output_.keys():
                output = output_[key]

                #

                try:
                    # get first predictions
                    tiled_predictions = output.T.cpu().numpy()
                    tiled_true = data[f'{self.simplex_names[key]}_y'].unsqueeze(0).cpu().numpy()

                    if self.virtual_idx is not None:
                        tiled_predictions = tiled_predictions[:, virtual[key]==0]
                        tiled_true = tiled_true[:, virtual[key]==0]

                    # Plot an image for logging
                    log_plot = plotting_func(tiled_predictions,
                                             tiled_true,
                                             )
                    # plot MAD
                    network_plot = self.plotter.plot_network(np.abs(tiled_predictions - tiled_true), data.wds_names,
                                                             )

                    # feature evolution plot
                    if hasattr(self.model, 'get_inspector') and 'layer_inspection' in self.writer.tables:
                        inspector = self.model.get_inspector(attr_to_dim[key])
                        features = [
                            src.surrogate_models.torch_models.visualization.plotter.plot_evolution_of_features(k) for k in inspector.functions_to_plot]
                        # add to the writer
                        self.writer.add_row(
                            'layer_inspection',
                            epoch,
                            f"Feature evolution {data.wds_names}",
                            *[self.writer.image(f) for f in features])

                    if hasattr(self, 'writer') and self.writer is not None:
                        self.writer.add_row(

                            str(self.plotter),
                            epoch,
                            f"Random Sample {data.wds_names}",
                            self.writer.image(log_plot),
                            self.writer.image(network_plot) if network_plot is not None else None,
                            None
                        )
                        # self.writer.save_table()

                    if self.show:
                        plt.show()
                    if self.save:
                        plt.savefig(f'{self.config.log_dir}/{key}_epoch_{epoch}-network_{data.wds_names}.png')
                    plt.close('all')
                except ValueError as e:
                    print(e)
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


class SimplexProfiler(SimplexTrainer):

    def __init__(self, profiler_schedule_args=None, *args, **kwargs, ):
        super().__init__(*args, **kwargs)
        self.profiler_schedule_args = profiler_schedule_args or {}

    def _train_epoch(self, epoch):
        with torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=self.profiler_schedule_args.get('wait', 1),
                    warmup=self.profiler_schedule_args.get('warmup', 1),
                    active=self.profiler_schedule_args.get('active', 2),
                    repeat=self.profiler_schedule_args.get('repeat', 1),
                ),
                on_trace_ready=tensorboard_trace_handler,
                with_stack=True
        ) as profiler:
            return super()._train_epoch(epoch)
