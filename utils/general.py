"""
Functions for object instantiation and value and types transformation.
"""

import importlib

import torch


def get_obj_from_str(string, reload=False):
    """
    Return the class selected by a string

    :param string: String with path to the *.py file and class name
    :param reload: Flag to reload object
    :return: Selected class
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """
    Wrapper for object instantiation by it name in the config

    :param config: OmegaConf object with
    :return: Instantiated object with provided parameters
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def q_normalize(q):
    """
    Normalize the coefficients of a given quaternion tensor of shape [*, 4].
    """
    assert q.shape[-1] == 4

    norm = torch.sqrt(torch.sum(torch.square(q), dim=-1))  # ||q|| = sqrt(w²+x²+y²+z²)
    assert not torch.any(
        torch.isclose(norm, torch.zeros_like(norm, device=q.device)))  # check for singularities
    return torch.div(q, norm[:, None])  # q_norm = q / ||q||


def s_act(x, min_s_value, max_s_value):
    if isinstance(x, float):
        x = torch.tensor(x).squeeze()
    return min_s_value + torch.sigmoid(x) * (max_s_value - min_s_value)


def s_inv_act(x, min_s_value, max_s_value):
    if isinstance(x, float):
        x = torch.tensor(x).squeeze()
    y = (x - min_s_value) / (max_s_value - min_s_value) + 1e-5
    y = torch.logit(y)
    assert not torch.isnan(
        y
    ).any(), f"{x.min()}, {x.max()}, {y.min()}, {y.max()}"
    return y


def o_act(x):
    if isinstance(x, float):
        x = torch.tensor(x).squeeze()
    return torch.sigmoid(x)


def o_inv_act(x):
    if isinstance(x, float):
        x = torch.tensor(x).squeeze()
    return torch.logit(x)
