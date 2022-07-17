from typing import Callable

from torch import nn

_Activation = Callable[[], nn.Module]
_Norm = Callable[[int], nn.Module]
