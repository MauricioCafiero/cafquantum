#!/bin/zsh
# cafquant runner: env guards + repo cwd, then the CLI.
# For long runs (big molecules, freq), launch detached per the machine-wide
# convention:
#   python -c "import subprocess,sys; subprocess.Popen(sys.argv[1:], stdout=open('out.log','ab'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)" zsh run_cafquant.sh <args>
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
caffeinate -w $$ &
cd "$(dirname "$0")"
./.venv/bin/python code/cafquant.py "$@"