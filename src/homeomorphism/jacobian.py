import re
from collections.abc import Iterable
from typing import Any, Callable

import torch
import torch.nn as nn

def remove_hooks(handles: Iterable[Any]) -> None:
	for handle in handles:
		handle.remove()


def _first_tensor_index(values: tuple[Any, ...]) -> int:
	for i, value in enumerate(values):
		if isinstance(value, torch.Tensor):
			return i
	raise ValueError("No tensor argument found in submodule inputs.")


def _to_tensor(value: Any) -> torch.Tensor:
	if isinstance(value, torch.Tensor):
		return value
	if isinstance(value, tuple) and value and isinstance(value[0], torch.Tensor):
		return value[0]
	raise TypeError("Submodule output is not tensor-like (Tensor or tuple[Tensor, ...]).")


def submodule_output_jacobian(
	model: nn.Module,
	module_path: str,
	model_inputs: dict[str, torch.Tensor],
	*,
	input_index: int | None = None,
	batch_index: int = 0,
	create_graph: bool = False,
	vectorize: bool = True,
) -> torch.Tensor:
	"""Compute d(submodule_output) / d(submodule_input) at the current forward point.

	The function runs one model forward pass to capture the target submodule call
	arguments, then computes a local Jacobian for that submodule invocation.
	"""
	target_module = model.get_submodule(module_path)
	captured_args: tuple[Any, ...] | None = None

	def capture_args(_mod: nn.Module, inp: tuple[Any, ...], _out: Any) -> None:
		nonlocal captured_args
		captured_args = inp

	handle = target_module.register_forward_hook(capture_args)
	try:
		_ = model(**model_inputs)
	finally:
		handle.remove()

	if captured_args is None:
		raise RuntimeError(f"Submodule '{module_path}' was not executed in forward pass.")

	index = input_index if input_index is not None else _first_tensor_index(captured_args)
	if index < 0 or index >= len(captured_args):
		raise IndexError(f"input_index {index} out of range for submodule inputs.")

	input_tensor = captured_args[index]
	if not isinstance(input_tensor, torch.Tensor):
		raise TypeError("Selected input is not a torch.Tensor.")
	if input_tensor.dim() == 0:
		raise ValueError("Selected input tensor must include a batch dimension.")
	if batch_index < 0 or batch_index >= input_tensor.shape[0]:
		raise IndexError(
			f"batch_index {batch_index} out of range for batch size {input_tensor.shape[0]}."
		)

	base_args = list(captured_args)
	x = input_tensor[batch_index : batch_index + 1].detach().clone().requires_grad_(True)

	for i, arg in enumerate(base_args):
		if isinstance(arg, torch.Tensor):
			if i == index:
				continue
			if arg.dim() > 0 and arg.shape[0] == input_tensor.shape[0]:
				base_args[i] = arg[batch_index : batch_index + 1].detach()
			else:
				base_args[i] = arg.detach()

	def local_forward(local_x: torch.Tensor) -> torch.Tensor:
		call_args = list(base_args)
		call_args[index] = local_x
		return _to_tensor(target_module(*call_args))

	return torch.autograd.functional.jacobian(
		local_forward,
		x,
		create_graph=create_graph,
		vectorize=vectorize,
	)
