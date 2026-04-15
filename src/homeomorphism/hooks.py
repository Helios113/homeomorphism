import re
from collections.abc import Iterable
from typing import Any, Callable

import torch
import torch.nn as nn


def get_submodule(model: nn.Module, module_path: str) -> nn.Module:
	"""Return a nested submodule by dotted path."""
	return model.get_submodule(module_path)

def get_submodules(model: nn.Module, prefix_path: str, items: list[str]) -> list[nn.Module]:
	"""Return a nested submodule by dotted path."""
	return [model.get_submodule(".".join([prefix_path, item_name])) for item_name in items]
	

def get_modules(model: nn.Module, items: list[str]) -> list[nn.Module]:
    names = list_module_names(model)
    module_paths = []
    for name in names:
        if name.split(".")[-1] in items:
            module_paths.append(name)
    return [model.get_submodule(module_path) for module_path in module_paths]
            

def list_module_names(model: nn.Module) -> list[str]:
	"""List all named modules to discover selectable sections."""
	return [name for name, _ in model.named_modules() if name]


def register_forward_capture_hooks(
	modules: Iterable[nn.Module],
	cache: dict[str, torch.Tensor],
	module_type : str,
	detach: bool = True,
	hook_fn: Callable[[nn.Module, tuple[Any, ...], Any, str, dict[str, torch.Tensor], bool], None] | None = None,
) -> list[Any]:
	"""Attach forward hooks and cache outputs by module name.

	Pass ``hook_fn`` to fully replace the default hook behavior.
	The provided callable receives ``(module, inputs, output, key, cache, detach)``.
	"""
	handles: list[Any] = []
	name_list = [f"{module_type}_{i}" for i, _ in enumerate(modules)]
	module_list = list(modules)

	for name, module in zip(name_list, module_list, strict=True):
		# default hook behaviour
		if hook_fn is None:
			def hook(_mod: nn.Module, _inp: tuple[Any, ...], out: Any, key: str = name):
				value = out[0] if isinstance(out, tuple) else out
				if isinstance(value, torch.Tensor):
					cache[key] = value.detach() if detach else value
		# custom hook pass through
		else:
			def hook(_mod: nn.Module, _inp: tuple[Any, ...], out: Any, key: str = name):
				hook_fn(_mod, _inp, out, key, cache, detach)

		handles.append(module.register_forward_hook(hook))

	return handles

