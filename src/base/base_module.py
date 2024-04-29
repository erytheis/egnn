import os
from os.path import dirname
from typing import overload


class BaseModule(object):
    module_root: str

    def handle_inputs(self, input: str):
        if os.path.isabs(input):
            return input
        elif '/' in input:
            return input
        elif input in self._possible_inputs:
            return os.path.join(self.input_dir, input)
        else:
            return input

    @classmethod
    def handle_local_inputs(cls, input: str):
        if os.path.isabs(input):
            return input
        else:
            return os.path.join(cls.input_dir, input)

    @property
    def _possible_inputs(self):
        return [None]

    @property
    def _input_foldername(self):
        return 'input'

    @property
    def input_dir(self):
        return os.path.join(self.module_root, self._input_foldername)

