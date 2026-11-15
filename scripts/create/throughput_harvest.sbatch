#!/bin/bash -l
# Throughput check on CREATE: IPPO on SocialJax Harvest (open), 7 agents, individual rewards (the dilemma),
# at the default 256 parallel envs. Two runs of 2 and 6 updates (1 update = 256 envs x 1000 steps);
# (time of 6 - time of 2) / 4 = seconds per update without compile time, so a 1e8-step run takes about
# 390 updates x that. Not a study.
#   sbatch ~/throughput_harvest.sbatch
# Output: /scratch/users/$USER/<jobid>.out
#SBATCH --job-name=sj_speed
#SBATCH --partition=interruptible_gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/users/%u/%j.out
set -euo pipefail

ROOT=/scratch/users/$USER/socialjax_work
# Some interruptible_gpu nodes do not carry this module name; the venv's python points straight at the
# /software build, so carry on without the module and stop only if that python does not run.
module load python/3.11.6-gcc-13.2.0 2>/dev/null || echo "module python/3.11.6 not found on $(hostname); using the venv python"
source "$ROOT/venv/bin/activate"
python -c "import sys; print('python', sys.version.split()[0], 'on', '$(hostname)')"
cd "$ROOT/SocialJax"
export PYTHONPATH=$PWD:${PYTHONPATH:-}

nvidia-smi --query-gpu=name,memory.total --format=csv
for UPDATES in 2 6; do
  STEPS=$((UPDATES * 256 * 1000))
  START=$(date +%s)
  python algorithms/train.py --algo IPPO --env harvest_open reward=individual \
    NUM_ENVS=256 TOTAL_TIMESTEPS=$STEPS WANDB_MODE=disabled SEED=1 > "/scratch/users/$USER/${SLURM_JOB_ID}_u${UPDATES}.log" 2>&1
  END=$(date +%s)
  echo "UPDATES=$UPDATES STEPS=$STEPS SECONDS=$((END - START))"
done
