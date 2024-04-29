import lovely_tensors as lt
import torch

import src.surrogate_models.torch_models.experiments as trainers
import src.surrogate_models.torch_models.loader.dataloaders as loaders
import src.surrogate_models.torch_models.model.loss as losses
import src.surrogate_models.torch_models.model.metric as all_metrics
import src.surrogate_models.torch_models.visualization.writer as writers
import src.surrogate_models.torch_models.model as models
from src.surrogate_models.torch_models.model.loss import CompositeLossFunction as clf
from src.surrogate_models.torch_models.parse_config import ConfigParser
from src.surrogate_models.torch_models.runners import load_datasets, load_args, load_cli_options
from src.utils.torch_utils import prepare_device

if __name__ == '__main__':
    import wandb
    # initial setup
    # mp.set_start_method('spawn')
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    # initial setup
    # mp.set_start_method('spawn')
    lt.monkey_patch()
    torch.cuda.empty_cache()
    # torch.autograd.set_detect_anomaly(True)
    # SEED = np.random.randint(0, 1e6)
    SEED = 90342  # good seed for BAK
    # SEED = 9031

    # parse config
    args = load_args('config.yaml')
    options = load_cli_options()
    config = ConfigParser.from_args(args, options)

    # init logging
    logger = config.get_logger('train')

    wandb.init()
    writer = config.init_obj('writer', writers, logger=logger, config=config)
    config = ConfigParser.from_sweep(wandb.config, config)
    logger.info(config.config)

    # init datasets
    device, device_ids = prepare_device(config['n_gpu'], config['device'])
    dataset = load_datasets(config, device)

    # setup loader instances
    data_loader = config.init_obj('loader', loaders, dataset=dataset)
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
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # # init writer

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

    experiments.run()
