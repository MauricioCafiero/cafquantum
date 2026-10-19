#!/bin/zsh
# cafquant environment setup: uv venv with all five backends + rdkit + ase
# torch 2.8.0 is the compatibility intersection:
#   fairchem-core 2.21 requires torch~=2.8.0, aimnet 0.2.0 requires >=2.8, mace-torch allows >=1.12
#
# e3nn note: mace-torch 0.3.16 hard-pins e3nn==0.4.4 (its Activation code does
# not work on 0.5), while fairchem-core declares e3nn>=0.5. In practice
# fairchem runs fine on 0.4.4, and cafquant.env patches the one real breakage
# (e3nn 0.4.4 torch.load vs torch>=2.6 weights_only default). So: pin 0.4.4,
# install BOTH fairchem-core and mace-torch with --no-deps and add their dep
# lists minus the e3nn pin.

cd "$(dirname "$0")"

# torch + tblite both load libomp on macOS -- same guards as run_cafquant.sh
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

/opt/homebrew/bin/uv venv --python 3.12 --allow-existing .venv

# shared core: torch stack, aimnet, tblite, ase, rdkit, plus the deps of
# fairchem-core and mace-torch (installed --no-deps below, minus their e3nn
# pins which we resolve ourselves)
/opt/homebrew/bin/uv pip install --python .venv/bin/python \
    "torch==2.8.0" \
    "numpy>=2.0,<2.5" \
    "e3nn==0.4.4" \
    "aimnet[ase]==0.2.0" \
    "tblite==0.7.0" \
    "ase==3.29.0" \
    "rdkit" \
    "opt_einsum" "torch-ema" "prettytable" "matscipy" "h5py" "torchmetrics" \
    "python-hostlist" "configargparse" "GitPython" "pyYAML" "tqdm" "lmdb" \
    "orjson" "matplotlib" "pandas" \
    "ase-db-backends>=0.10.0" "clusterscope==0.0.18" "huggingface-hub>=0.27.1" \
    "hydra-core" "monty>=2026.2.18" "numba>=0.62.0" \
    "ray[serve]>=2.53.0" "requests" "scipy>=1.15.0" "setuptools<81.0.0" \
    "submitit" "torchtnt" "websockets" "wandb"

/opt/homebrew/bin/uv pip install --python .venv/bin/python --no-deps \
    "fairchem-core==2.21.0" "mace-torch==0.3.16"

echo "--- import probe ---"
cd code
../.venv/bin/python -c "
import cafquant.env
import torch, fairchem.core, mace.calculators, aimnet.calculators, tblite.ase, ase, rdkit
print('torch', torch.__version__, '| ase', ase.__version__, '| all five backends import OK')
" 2>&1 | grep -v "NOTE: Redirects\|Cuequivariance\|cuequivariance"