"""cafquant: multi-MLIP + semiempirical workspace (UMA, MACE-OFF, Egret-1, AIMNet2,
GFN2-xTB, PM7).

All six backends are exposed as ASE calculators; sp/opt/freq run through the
same ASE driver for every method. Import :mod:`cafquant.env` before anything
that touches torch.
"""
from . import env  # noqa: F401  (must come before any torch import)

__version__ = "0.1.0"

METHODS = ("uma", "maceoff", "egret", "aimnet2", "gfn2", "pm7")