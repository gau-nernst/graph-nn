from typing import Callable

from torch import nn

__all__ = ["_Activation", "_Norm"]

_Activation = Callable[[], nn.Module]
_Norm = Callable[[int], nn.Module]
