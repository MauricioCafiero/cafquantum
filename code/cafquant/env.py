"""Environment guards for cafquant. Import this module FIRST, before any torch import.

KMP_DUPLICATE_LIB_OK=TRUE is needed on macOS because torch and tblite both load
OpenMP runtimes (libomp vs libgomp). Even the guards are not enough to mix the
two in ONE process (pthread_mutex_init EINVAL crash), so the CLI runs torch-
backed and non-torch methods in separate worker subprocesses -- this module
stays torch-free.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def patch_torch_for_e3nn() -> None:
    """e3nn 0.4.4 (hard-pinned by mace-torch 0.3.16) loads o3/constants.pt with
    a plain torch.load, which under torch>=2.6's weights_only default raises
    WeightsUnpicklerError unless "slice" is allow-listed. Call before importing
    fairchem/mace (both import e3nn)."""
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([slice])
    except Exception:
        pass