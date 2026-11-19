#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/independent_140_v04
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python run_independent.py --chunk --start 0 --stop 500
python run_independent.py --chunk --start 500 --stop 1000
python run_independent.py --finalize
