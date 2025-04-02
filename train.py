import matplotlib
import torch

import src.surrogate_models.torch_models.experiments as trainers
import src.surrogate_models.torch_models.experiments.schedulers as schedulers
import src.surrogate_models.torch_models.loader.dataloaders as loaders
import src.surrogate_models.torch_models.model as models
import src.surrogate_models.torch_models.model.loss as losses
import src.surrogate_models.torch_models.model.metric as all_metrics
import src.surrogate_models.torch_models.visualization.writer as writers
from src.surrogate_models.torch_models.model.loss import CompositeLossFunction as clf
from src.surrogate_models.torch_models.parse_config import ConfigParser
from src.surrogate_models.torch_models.runners import load_datasets, load_args, load_cli_options
from src.utils.torch_utils import prepare_device

if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    # initial setup
    torch.multiprocessing.set_start_method('spawn')
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)
    SEED = 90342

    # parse config
    args = load_args(
        'config_zj_3.2b.yaml',
    )

    options = load_cli_options()
    config = ConfigParser.from_args(args, options)

    # init logging
    logger = config.get_logger('train')
    device, device_ids = prepare_device(config['n_gpu'], config['device'])

    # init datasets
    dataset = load_datasets(config, device)

    # setup loader instances
    data_loader = config.init_obj('loader', loaders, dataset=dataset,loaded_indices=None)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', models)
    model = model.to(device)

    # init params for logging
    with torch.no_grad():
        for dummy_batch in data_loader:
            dummy_out = model.forward(dummy_batch.to(device))
            break
    logger.info(model)

    # prepare for (multi-device) GPU training
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = clf([getattr(losses, l['type'])(dataset=dataset, **l["args"]) for l in config['loss']])
    metrics = [getattr(all_metrics, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', schedulers, optimizer)

    # init writer
    writer = config.init_obj('writer', writers, logger=logger,
                             layer_inspection_functions=None,
                             config=config) if not config['debug'] else None

    experiments = config.init_obj('trainer', trainers,
                                  model=model,
                                  criterion=criterion,
                                  metric_ftns=metrics,
                                  optimizer=optimizer,
                                  writer=writer,
                                  config=config,
                                  device=device,
                                  data_loader=data_loader,
                                  valid_data_loader=valid_data_loader,
                                  lr_scheduler=lr_scheduler)
    if not config['debug']:
        experiments.run()
