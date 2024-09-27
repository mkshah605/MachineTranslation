import torch as t
import torch.nn as nn


def beam_search(model: nn.Module, max_depth: int, n_beams: int) -> t.Tensor:
    raise NotImplementedError