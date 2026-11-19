# 140B auxiliary-42 parity reselection experiment

**Status:** Completed on a synthetic hard optical channel; comparison 928/1000 vs 939/1000, auxiliary CRC failures 13 vs 6. This is not a QR/JAB or real-camera benchmark.

## Audit existing artifacts (fast)

Run from the root of the extracted package:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_outputs.py
```

The file `RESEARCH_REPORT_JA.md` contains (a)-(e), protocol, exact paired statistics, conditions and limitations. `commands_executed.txt` records the shell commands. `PROTOCOL_PRE_REGISTERED.md` is the pre-selection decision rule and must not be edited to reinterpret results.

## Recompute from scratch (development and final validation)

Dependencies: Python 3.13, NumPy 2.3+, SciPy 1.15+, OpenCV 4.10+. `optical_code_adversarial_v02/` is included from the original attached ZIP; its original source files are not rewritten by the new code. Scripts write only under `outputs/` except reference `final_validation.py`, which overwrites its own reference results in the extracted copy.

```bash
cd optical_code_adversarial_v02
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python final_validation.py --condition hard --name advgraph140 --n 1000
cd ..
# Reset only generated development outputs if redoing from zero; keep archived baseline + protocol.
rm -f outputs/{screen_dev.csv,selection_dev.csv,selected_aux42_rows.npy,frozen_selection.json,frozen_holdout_summary.json,holdout_pair_*.csv}
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python aux42_search.py --stage analytic
for bounds in '0 7' '7 14' '14 16'; do set -- $bounds; OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python aux42_search.py --stage screen --start "$1" --stop "$2"; done
for bounds in '0 2' '2 4' '4 6'; do set -- $bounds; OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python aux42_search.py --stage select --start "$1" --stop "$2"; done
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python aux42_search.py --stage freeze
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python aux42_holdout.py --stage chunk --start 0 --stop 500
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python aux42_holdout.py --stage chunk --start 500 --stop 1000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python aux42_holdout.py --stage finish
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_outputs.py
```

`rm` above is for a **copy** of the archive; keep the provided raw outputs immutable for audit. Python/NumPy RNG ABI and OpenCV camera processing changes may alter synthetic numerical results. Changing model selection or checking final results between candidate choices invalidates a future independent holdout claim.

## Notes

- The old holdout has been published before and therefore is not truly fresh to the scientific project. Selection of the new row matrix did not inspect any per-item final outcomes; see frozen SHA256 and independent developer seeds.
- GitHub clone failed in this runtime because DNS for github.com failed. The original attached ZIP was used; `robust_code.py` and `final_validation.py` Git blob hashes match commit `4091ba8`.
- New package does not modify the public repository. All scripts and files are exported only as downloadable artifacts.
