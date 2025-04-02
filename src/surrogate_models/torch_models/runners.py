import argparse
import collections
import distutils
import os
import warnings

import torch
import torch_geometric

import src.surrogate_models.torch_models.dataset as all_datasets
import src.surrogate_models.torch_models.dataset.transforms as all_transforms
import src.surrogate_models.torch_models.experiments as trainers
import src.surrogate_models.torch_models.loader.dataloaders as loaders
import src.surrogate_models.torch_models.model as models
import src.surrogate_models.torch_models.model.loss as losses
import src.surrogate_models.torch_models.model.metric as all_metrics
import src.surrogate_models.torch_models.visualization.writer as writers
# from src.surrogate_models.torch_models.base.base_dataset import ConcatDataset
from src.surrogate_models.torch_models.base.base_dataset import ConcatDataset
from src.surrogate_models.torch_models.dataset.base_gnn_dataset import concat_gnn_datas
from src.surrogate_models.torch_models.model.loss import CompositeLossFunction as clf
from src.utils.torch_utils import prepare_device
from src.utils.utils import all_equal


# from src.surrogate_models.torch_models.data.complex_data import Cochain



def load_transforms_from_memory(config):
    checkpoint = config._config.get('resume', None)
    if checkpoint is None:
        return None, None
    checkpoint = os.path.dirname(checkpoint)

    # load saved transforms
    filename = os.path.join(checkpoint, 'transform.pth')
    try:
        loaded_transforms = torch.load(filename)
    except FileNotFoundError:
        warnings.warn(f'No transforms found at {filename}')
        loaded_transforms = None

    # load saved pre_transforms
    filename = os.path.join(checkpoint, 'pre_transform.pth')
    try:
        loaded_pre_transforms = torch.load(filename)
    except FileNotFoundError:
        warnings.warn(f'No pre_transforms found at {filename}')
        loaded_pre_transforms = None
    return loaded_transforms, loaded_pre_transforms


def load_datasets(config, device, override_transforms=None, override_pre_transforms=None, pre_transform_kwargs={}):
    dataset, transforms, pre_transforms = [], None, None
    combined_override_pre_transforms, combined_override_transforms = None, None
    dataset_types = []

    transforms = config['dataset'].get('transforms', [])
    pre_transforms = config['dataset'].get('pre_transforms', [])


    # firstly repack transforms into a single list
    individual_transform = []
    for transform in transforms:
        if not transform.get('combined', False):
            individual_transform.append(transform)

    # repack pre_transforms into a single list
    individual_pre_transform, augments = [], []
    for transform in pre_transforms:
        combined = transform['combined'] if 'combined' in transform.keys() else False
        if not combined:
            individual_pre_transform.append(transform)

    # add common args to all subsets
    common_args = config['dataset']['args'] if 'args' in config['dataset'] else {}

    for ds in config['dataset']["subsets"]:
        # assign pre transforms to each dataset
        pre_tr = ds.get('pre_transforms', [])
        pre_tr.extend(individual_pre_transform)

        # pre transforms
        pre_transforms = all_transforms.Compose([getattr(all_transforms, transform["type"])(**transform["args"],
                                                                                            # **pre_transform_kwargs.get(i)
                                                                                            )
                                                 for i, transform in enumerate(pre_tr)])

        # assign transforms to each dataset
        tr = ds.get('transforms', [])
        tr.extend(individual_transform)

        transforms = all_transforms.Compose([getattr(all_transforms, transform["type"])(**transform["args"])
                                             for transform in tr])


        subset = getattr(all_datasets, ds['type'])(**{**common_args, **ds['args']},
                                                   pre_transform=pre_transforms if override_pre_transforms is None else override_pre_transforms,
                                                   transform=transforms if override_transforms is None else override_transforms)

        subset.data_to(device)
        dataset.append(subset)

        # place in the corresponding subset
        key = 'train'
        key = 'validation' if ds.get('validation_only', False) else key
        key = 'test' if ds.get('test_only', False) else key
        dataset_types.append(key)

    # concat datasets
    if len(dataset) > 1:
        assert all_equal([type(d) for d in dataset]), "Dataset types must be equal for concatenation"
        dataset = ConcatDataset(dataset)
        dataset.types = dataset_types
    else:
        dataset = dataset[0]

    if override_transforms is not None and override_pre_transforms is not None:
        for ds in dataset.datasets:
            # add loaded combined transforms to the dataset
            if combined_override_pre_transforms is not None:
                combined_override_pre_transforms(ds.data)
                [subset.pre_transform.insert(0, c) for c in combined_override_pre_transforms.transforms[::-1]]
            if combined_override_transforms is not None:
                combined_override_transforms(ds.data)
                [subset.transform.insert(0, c) for c in combined_override_transforms.transforms[::-1]]
        return dataset

    data = concat_gnn_datas([d.data for d in dataset.datasets])

    # infer parameters for transforms im combined data such as normalization parameters etc.
    combined_transforms = all_transforms.Compose(
        [getattr(all_transforms, transform["type"])(**transform.get("args", {}))
         for transform in config['dataset'].get('transforms', [])
         if 'combined' in transform.keys()])

    combined_pre_transforms = all_transforms.Compose(
        [getattr(all_transforms, transform["type"])(**transform.get("args", {}))
         for transform in config['dataset'].get('pre_transforms', [])
         if 'combined' in transform.keys()])

    # deal with complex data obj
    if isinstance(data[0], dict):
        datas = {i: d for i, d in enumerate(data)}
        tr = {dim: combined_transforms.copy().infer_parameters(d) for dim, d in datas.items()}
        pre_tr = {dim: combined_pre_transforms.copy().infer_parameters(d) for dim, d in datas.items()}
    else:
        [combined_pre_transforms.infer_parameters(d) for d in data]

    for ds in dataset.datasets:
        # apply pre_transforms
        ds._data_list = None
        ds.data = combined_pre_transforms(ds.data)

        if isinstance(ds.transform, torch_geometric.transforms.BaseTransform):
            [ds.pre_transform.insert(0, c) for c in combined_pre_transforms.transforms[::-1]]
            [ds.transform.insert(0, c) for c in combined_transforms.transforms[::-1]]
        elif isinstance(ds.transform, dict):
            [ds.transform[k].extend(0, t) for k, t in tr.items()]
            [ds.pre_transform[k].extend(0, t) for k, t in pre_tr.items()]
        else:
            ds.transform = combined_transforms
            ds.pre_transform = combined_pre_transforms

        ds.extract_subgraphs() if hasattr(ds, 'extract_subgraphs') else None

    # infer parameters on the combined dataset. Needed for normalization etc.
    data = concat_gnn_datas([d.data for d in dataset.datasets])
    [combined_transforms.infer_parameters(d) for d in data]
    return dataset


def load_args(config_name=None, resume=None, device=None, SEED=90342):
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=config_name, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=device, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--seed', default=SEED, type=int)
    args.add_argument('-r', '--resume', default=resume, type=str)
    args.add_argument('--debug', default=None, type=bool)

    # boolean vlaues with both options
    parsed_bool = lambda x: bool(distutils.util.strtobool(x))
    args.add_argument('--re', default=None, type=parsed_bool)
    args.add_argument('--or', default=None, type=parsed_bool)

    return args


def load_cli_options():
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='loader;args;batch_size'),
        CustomArgs(['--crp', '--cache_reset_period'], type=int, target='trainer;args;cache_clear_period'),
        CustomArgs(['--or', '--optimizer_reset'], type=bool, target='optimizer;reset'),
        CustomArgs(['--re', '--reload_data'], type=bool, target='dataset;args;reload_data')
    ]
    return options


def load_experiment(config):
    logger = config.get_logger('train')

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

    # test_headloss(data_loader)
    # init writer

    writer = config.init_obj('writer', writers, logger=logger, config=config) if not config['debug'] else None

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
    return experiments
