#!/usr/bin/env bash
set -euo pipefail

python -m hermes.experiments.aggregate --results results/qemfi_m0
python -m hermes.experiments.plot --results results/qemfi_m0

