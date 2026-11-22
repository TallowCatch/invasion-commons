#!/bin/bash -l
# OPTIONAL alternative (22 Nov): the LLM inspector ladder with vLLM on CREATE GPUs. The main path is Ollama
# (notes/llm_overseer/README.md); this script is UNTESTED. It judges the same 600-item subset with one HuggingFace
# model per array task (Qwen2.5-Instruct, bf16).
#
# One-off setup, in an interactive CPU session (compute nodes have internet; the login node is too small):
#   srun -p cpu --mem=16G --cpus-per-task=4 --time=01:00:00 --pty /bin/bash -l
#   module load python/3.11.6-gcc-13.2.0
#   python -m venv /scratch/users/$USER/vllm_venv && source /scratch/users/$USER/vllm_venv/bin/activate
#   pip install "vllm==0.32.0" numpy        # needs Python >= 3.11 and glibc >= 2.28; if the GPU driver is too old for
#                                           # its torch, try "vllm==0.11.0" (torch 2.8; Qwen2.5 only)
#   export HF_HOME=/scratch/users/$USER/hf_cache
#   for m in 0.5B 1.5B 3B 7B 14B 32B; do huggingface-cli download Qwen/Qwen2.5-$m-Instruct; done   # about 110 GB
# Data: the Harvest set is in git; copy the BazaarBench set (not in git) from the laptop:
#   scp data/bazaarbench/llm_judge/bazaar_judge.jsonl.gz create:$REPO/data/bazaarbench/llm_judge/
# Submit (array index = model; 32B needs 2 GPUs or one 80 GB card):
#   sbatch --array=1-5 scripts/create/llm_judge.sbatch
#   sbatch --array=6 --gpus=2 scripts/create/llm_judge.sbatch
#SBATCH --job-name=llm_judge
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/%u/llm_overseer/logs/%A_%a.out
set -euo pipefail

MODELS=(Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct
        Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct)
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID - 1))]}
module load python/3.11.6-gcc-13.2.0 2>/dev/null || true
source /scratch/users/$USER/vllm_venv/bin/activate
export HF_HOME=/scratch/users/$USER/hf_cache HF_HUB_OFFLINE=1
cd "${REPO:-/scratch/users/$USER/socialjax_work/invasion-commons}"
export PYTHONPATH=$PWD
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
OUT=/scratch/users/$USER/llm_overseer/judge_vllm
for MODE in direct reason; do
  python -m experiments.llm_overseer.run_judge --backend vllm --model "$MODEL" --mode "$MODE" --out-dir "$OUT"
done
