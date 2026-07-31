r"""Custom functions to register by Hydra"""
from typing import Callable, Dict

from omegaconf import OmegaConf


def register_custom_resolvers(extra_resolvers: Dict[str, Callable] = None):
    """Wrap your main function with this.
    You can pass extra kwargs, e.g. `version_base` introduced in 1.2.
    """
    extra_resolvers = extra_resolvers or {}
    for name, resolver in extra_resolvers.items():
        OmegaConf.register_new_resolver(name, resolver)

def effective_lr(base_lr: float, batch_size: int, n_GPUs: int, n_nodes: int, contrastive_setting: str) -> float:
    if contrastive_setting == "shuffle_negative":
        return base_lr * batch_size* batch_size * n_GPUs * n_nodes / 256
    else: # "withinbatch_negative" or "collapse_sequence" settings, which do not create extra negatives
        return base_lr * batch_size * n_GPUs * n_nodes / 256

def register_resolvers():
    register_custom_resolvers({
        "eval": eval,
        "effective_lr": effective_lr,
        "len": len
    })