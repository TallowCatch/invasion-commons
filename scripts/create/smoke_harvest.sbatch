#!/bin/bash -l
# Smoke test on CREATE: a tiny IPPO run on SocialJax Harvest (open), 7 agents, no wandb.
# It checks that JAX sees the GPU and the training loop runs end to end; the result is not a study.
#   sbatch ~/smoke_harvest.sbatch
# Output: /scratch/users/$USER/<jobid>.out
#SBATCH --job-name=sj_smoke
#SBATCH --partition=interruptible_gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/users/%u/%j.out
set -euo pipefail

ROOT=/scratch/users/$USER/socialjax_work
module load python/3.11.6-gcc-13.2.0
source "$ROOT/venv/bin/activate"
cd "$ROOT/SocialJax"
export PYTHONPATH=$PWD:${PYTHONPATH:-}

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import jax; print('JAX devices:', jax.devices())"

# 8 envs x 1000 steps per update; 2e5 steps = 25 updates. Minibatches: 8*1000*7/500 = 112 per minibatch.
time python algorithms/train.py --algo IPPO --env harvest_open \
  NUM_ENVS=8 TOTAL_TIMESTEPS=200000 WANDB_MODE=disabled SEED=1
