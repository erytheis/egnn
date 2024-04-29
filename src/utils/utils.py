import errno
import json
import os
import os.path as osp
from itertools import groupby
from os import listdir
from os.path import isfile, join, isabs
from typing import List, Iterable

import numpy as np
import scipy.sparse as sp
import yaml
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.utils import to_scipy_sparse_matrix

PROJECT_ROOT: bytes = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
RUN_CHECKS: bool = False
DEBUG = False

# VISUALIZATION UTILS
DEFAULT_COLORS = ["#ffbe0b","#fd8a09","#fb5607","#fd2b3b","#ff006e","#d61398","#c11cad","#8338ec","#5f5ff6","#3a86ff"]
DEFAULT_COLORS_2 = ["#f26419","#E88044","#de9b6f","#cad2c5","#84a98c","#52796f","#354f52"]
DEFAULT_COLORS_2_ADDITIONAL = ["#D81159"]
DEFAULT_BG_COLOR = "#E7EFED"


def set_debug():
    global DEBUG
    DEBUG = True

def get_abs_path(path):
    if isabs(path):
        return path
    else:
        return join(PROJECT_ROOT, path)


def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def list_filenames(mypath) -> List[str]:
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def read_yaml(path):
    with open(path, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data


def write_yaml(data, path):
    with open(path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def init_and_fit_ohe(data_list):
    enc = OneHotEncoder()
    return enc.fit(np.array(data_list).reshape(-1, 1))


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)



def num_connected_components(edge_index, num_nodes, connection='weak'):
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

    num_components, component = sp.csgraph.connected_components(
        adj, connection=connection)

    return num_components, component

def __repr__(self) -> str:
    return f'{self.__class__.__name__}({self.num_components})'


class Iterator:
    """
    Iterator for the compose objects.
    """
    __index: int
    __iterable: Iterable

    def __init__(self, obj):
        self.__iterable = obj.iterable
        self.__index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.__index >= len(self.__iterable):
            raise StopIteration

        # return the next color
        item = self.__iterable[self.__index]
        self.__index += 1
        return item

    def __len__(self):
        return len(self.__iterable)

    def __getitem__(self, index):
        return self.__iterable[index]  # return the item at the index

    def __contains__(self, item):
        return item in self.__iterable




from tqdm.auto import tqdm
from joblib import Parallel


class ProgressParallel(Parallel):
    """A helper class for adding tqdm progressbar to the joblib library."""
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def flatten_list(list_of_lists):
    """
    Flatten a list of lists into a single list.
    """
    return [item for sublist in list_of_lists for item in sublist]


def path_without_overwriting(path):
    extension = os.path.splitext(path)[1]
    path = os.path.splitext(path)[0]
    if os.path.exists(path + extension):
        i = 1
        while os.path.exists(path + f'_{i}' + extension):
            i += 1
        path = path + f'_{i}' + extension
    return path


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
