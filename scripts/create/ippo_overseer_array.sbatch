#!/bin/bash -l
# Phase 2 on CREATE: one IPPO + overseer run per array task (notes/socialjax/phase2_ippo/protocol.md).
# RUNS is a TSV from experiments/socialjax/phase2_runs.py; task i runs line i (1-based).
#   sbatch --array=1-48%8 --export=ALL,RUNS=$HOME/ippo/runs_stage1.tsv ~/ippo/ippo_overseer_array.sbatch
# public_gpu allows 8 GPUs per user, hence %8. Output: /scratch/users/$USER/ippo_overseer/<run name>/
#SBATCH --job-name=ippo_ov
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/users/%u/ippo_overseer/logs/%A_%a.out
set -euo pipefail

WORK=/scratch/users/$USER/socialjax_work
module load python/3.11.6-gcc-13.2.0 2>/dev/null || echo "module python/3.11.6 not found on $(hostname); using the venv python"
source "$WORK/venv/bin/activate"
cd "$WORK/invasion-commons"
export PYTHONPATH=$PWD:$WORK/SocialJax:${PYTHONPATH:-}

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$RUNS")
NAME=$(cut -f1 <<< "$LINE")
ARGS=$(cut -f2 <<< "$LINE")
OUT=/scratch/users/$USER/ippo_overseer/$NAME
if [ -f "$OUT/summary.json" ]; then echo "$NAME already done"; exit 0; fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# 16 Nov: on arc-hpc-comp035 no GPU was visible and JAX silently ran on the CPU until the 8 h limit. Fail fast instead.
python -c "import jax, sys; b = jax.default_backend(); print('jax backend:', b); sys.exit(b != 'gpu')" \
    || { echo "no GPU visible on $(hostname); stopping"; exit 1; }
echo "$NAME: $ARGS"
python -m experiments.socialjax.train_ippo_overseer --out "$OUT" $ARGS
