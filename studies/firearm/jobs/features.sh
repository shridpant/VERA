#!/bin/bash
# ImageBind segment features (LAVAD procedure) for every video of the finished A/G/AG run, on one GPU.
# Usage (on Oz, from launch.sh with an absolute path): bash studies/firearm/jobs/features.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
IB=${IB:-/local/datasets/sp2117/firearm_visilab/envs/imagebind/bin/python}
export PYTHONUTF8=1
$IB -B run_firearm_study.py features --config studies/firearm/experiment_A_G_AG.json
