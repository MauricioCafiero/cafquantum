"""cafquant: multi-MLIP + semiempirical workspace (UMA, MACE-OFF, Egret-1,
AIMNet2, GFN2-xTB, PM7, MACE-POLAR-1, MACE-OMOL-0, UMA-1.2).

All backends are exposed as ASE calculators; sp/opt/freq run through the
same ASE driver for every method. Import :mod:`cafquant.env` before anything
that touches torch.
"""
from . import env  # noqa: F401  (must come before any torch import)

__version__ = "0.1.0"

METHODS = ("uma", "maceoff", "egret", "aimnet2", "gfn2", "pm7",
           "polar1-s", "polar1-m", "omol", "uma1p2")

# what the CLI runs when -m/--method/--all are all omitted
DEFAULT_METHOD = "gfn2"