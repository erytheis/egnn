import types
from functools import wraps



def watch_variable(func):
    @wraps(func)
    def wrapper(self, layer_id=None, variable=None, *args, **kwargs):
        result = func(self, layer_id, variable, *args, **kwargs)

        if not hasattr(self, 'layer_inspector'):
            return result
        elif not self.layer_inspector.active:
            return result
        if type(result) == tuple:
            to_record = (r.clone().detach().requires_grad_(True) if hasattr(r, 'clone') else r for r in result)
        elif hasattr(result, 'clone'):
            to_record = result.clone().detach().requires_grad_(True)
        else:
            to_record = None

        name = str(func.__name__)
        self.layer_inspector.watch_variable(name, layer_id, to_record, *args, **kwargs)
        return result

    return wrapper


class LayerInspector:

    def add_to_watchlist(self, func_name):
        self.watch_dict[func_name] = {}

    def __init__(self):
        self.watch_dict = {}
        self.active = False
        self.functions_to_plot = []

    def watch_variable(self, func_name, layer_id, variable, *args, **kwargs):
        self.watch_dict[func_name][layer_id] = variable

    # def plot_evolution_of_features(self,function_to_plot):
    #     features = self.watch_dict[function_to_plot]
    #     return plot_evolution_of_features(features)


class LayerInspectorMeta(type):

    def __new__(mcls, name, bases, body):
        # assign a layer inspector object to the class
        body['layer_inspector'] = LayerInspector()
        # create new class object
        for name, obj in body.items():
            if name[:2] == name[-2:] == '__':
                # skip special method names like __init__
                continue
            if name in ['init_conv', 'reset_parameters',
                        'layer_inspector', 'forward', 'process',
                        'decode', 'encode', 'check_convergence']:
                # skip methods that are not called by the model
                continue
            if isinstance(obj, types.FunctionType):
                # decorate all functions
                body[name] = watch_variable(obj)
                body['layer_inspector'].add_to_watchlist(name)

        # same for the BaseGNN parent
        base_gnn = [b for b in bases if b.__name__ == 'BaseGNN']
        if len(base_gnn) > 0:
            for name, obj in base_gnn[0].__dict__.items():
                if name[:2] == name[-2:] == '__':
                    # skip special method names like __init__
                    continue
                if name in ['init_conv', 'reset_parameters', 'layer_inspector', 'forward']:
                    # skip methods that are not called by the model
                    continue
                if isinstance(obj, types.FunctionType):
                    # decorate all functions
                    body[name] = watch_variable(obj)
                    body['layer_inspector'].add_to_watchlist(name)

        return super(LayerInspectorMeta, mcls).__new__(mcls, name, bases, body)

    def __call__(cls, *args, **kwargs):
        # create a new instance for this class
        instance = super(LayerInspectorMeta, cls).__call__(*args, **kwargs)
        return instance


def get_evolution_of_features(layer_insector, key, vmax=None, vmin=None):
    feature_dict = layer_insector.watch_dict[key]


