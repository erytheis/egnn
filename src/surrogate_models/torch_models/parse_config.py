import argparse
import os
import logging
from copy import deepcopy
from os.path import dirname, join
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime

import numpy as np
import torch
import yaml
from torch_geometric import seed_everything

from src.base.base_module import BaseModule
from src.surrogate_models.torch_models.logger import setup_logging
from src.utils.utils import get_abs_path, read_json, write_json, read_yaml, write_yaml, PROJECT_ROOT, DEBUG


class ConfigParser(BaseModule):
    module_root = PROJECT_ROOT
    input_dir = join(module_root, 'input')

    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)

        # add args key if not existent and define default values
        add_default_args(self.config)
        self._config['inspect'] = self.config.get('inspect', False)

        # override resume from the config file
        resume = config.get('resume', resume)
        self.resume = get_abs_path(resume) if resume else None


        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

        # set save_dir where trained model and log will be saved.
        if 'trainer' in self.config.keys():
            experiment = 'trainer'
        elif 'tester' in self.config.keys():
            experiment = 'tester'
        else:
            return

        # finally
        base_dir = Path(get_abs_path(self.config[experiment]["args"]['save_dir']))
        exper_name = self.config['name']
        if run_id is None:  # use timestamp as default run-id
            self.run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        else:
            self.run_id = run_id
        self._save_dir = base_dir / 'models' / exper_name / self.run_id
        self._log_dir = base_dir / 'log' / exper_name / self.run_id
        self.base_dir = base_dir

            # configure logging module

    def mkdirs(self):
        # make directory for saving checkpoints and log.
        exist_ok = self.run_id == ''
        try:
            self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
            self.log_dir.mkdir(parents=True, exist_ok=exist_ok)
        except FileExistsError as r:
            print("Warning", r)

            # save updated config file to the checkpoint dir
        write_yaml(self.config, self.save_dir / 'config.yaml')

    @classmethod
    def from_dict(cls, config, *args, **kwargs):

        return cls(config, *args, **kwargs)

    @classmethod
    def from_args(cls, args, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            try:
                args.add_argument(*opt.flags, default=None, type=opt.type)
            except argparse.ArgumentError:
                continue
        if not isinstance(args, tuple):
            args, unknown = args.parse_known_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        if args.resume is not None:
            resume = Path(args.resume)
            resume_cfg_fname = resume.parent / 'config.yaml'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.yaml', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_path = cls.handle_local_inputs(args.config)
            resume_cfg_fname = Path(cfg_path)

        try:
            config = read_yaml(get_abs_path(resume_cfg_fname))
            config['resume'] = str(resume) if resume is not None else None
            print('Updating config from the checkpoint...')
        except FileNotFoundError as e:
            print('Warning: ', e)
            config = {}

        if resume:
            # update new config for fine-tuning
            # config.update(read_yaml(args.config))
            if args.config:
                config.update(read_yaml(cls.handle_local_inputs(args.config)))
            config['resume'] = str(resume)

        if args.seed is not None:
            SEED = args.seed
            print('seed set to', SEED)
            seed_everything(SEED)
            config.update({'seed': SEED})

        if args.debug:
            DEBUG = True
            print('DEBUG set to', DEBUG)
            config.update({'debug': True})

        if args.device is not None:
            config['device'] = args.device
        elif 'device' not in config.keys():
            config['device'] = 'cpu'

        # parse custom cli options into dictionary
        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}

        return cls(config, resume, modification)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        if name in self.config.keys():
            module_name = self[name]['type']
        else:
            return None
        module_args = dict(self[name]['args'])
        # assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'

        # check if there are kwargs to load from self.resume
        if self.resume is not None:
            resume_path = os.path.dirname(self.resume)
            if f'{name}_kwargs.json' in os.listdir(resume_path):
                resume_kwargs = read_json(os.path.join(resume_path, f'{name}_kwargs.json'))
                module_args.update(resume_kwargs)

        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def __contains__(self, item):
        return item in self.config

    def __setitem__(self, key, value):
        """Access items like ordinary dict."""
        self.config[key] = value

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                       self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    def update_with_sweep(self, sweep_config):
        for key, value in sweep_config.items():
            if '.' in key:
                # print(key,value)
                _set_by_path(self._config, key, value, separator='.')
                # print(_get_by_path(self, key.split('.')))
            else:
                self._config[key] = value

    @classmethod
    def from_sweep(cls, sweep_config, old_config_parser, *args, **kwargs):
        old_config = deepcopy(old_config_parser.config)
        print(sweep_config)
        for key, value in sweep_config.items():
            if '.' in key:
                _set_by_path(old_config, key, value, separator='.')
            else:
                old_config[key] = value
        return cls(old_config, *args, **kwargs)


def add_default_args(config):
    # res['non-leaf'] += 1
    nodes = config.keys()
    nothing_found = True
    for node in nodes:
        subnode = config[node]
        if isinstance(subnode, dict):
            if node != 'args':
                nothing_found = False
                add_default_args(subnode)
        if isinstance(subnode, list):
            for i, item in enumerate(subnode):
                if isinstance(item, dict):
                    nothing_found = False
                    add_default_args(item)
    if nothing_found and 'args' not in config:
        config['args'] = {}


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value, separator=';'):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(separator)
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _getitem(a,b):
    "Same as a[b] but also works for lists"
    if isinstance(a, list):
        return a[int(b)]
    return a[b]


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(_getitem, keys, tree)


def load_test_config():

    PROJECT_ROOT
    # parse config
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=join(PROJECT_ROOT, 'test', 'test.yaml'), type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume',
                      # default='/cluster/home/bulatk/cluster/home/bulatk/tmp/pythonProject/saved/training_logs/models/NormalizedModels/0627_101516/checkpoint-epoch400-loss0.0026.pth',
                      default=None,
                      type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='cpu', type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--seed', default=4242, type=int)

    # custom cli options to modify configuration from default values given in json file.
    return ConfigParser.from_args(args)
    # init logging
