#!/usr/bin/env bash
set -euo pipefail

python -m hermes.experiments.run_sweep --config configs/experiment/qemfi_m0_main.yaml

