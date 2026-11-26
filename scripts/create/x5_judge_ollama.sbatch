#!/bin/bash -l
# X5, the capability curve (notes/formulation/x5_capability/predictions.md): one Qwen2.5-Instruct judge on the
# 2,463-item stratified bank, raw-map prompt, --mode reason, max tokens 2048, served by Ollama on one GPU node.
# No cloud model and no paid API: the Ollama binary and the weights live on scratch.
#
# One-off setup (done 26 Nov 2026, login node):
#   OLL=/scratch/users/$USER/ollama: Ollama v0.40.16 (ollama-linux-amd64.tar.zst from the GitHub release),
#   models pulled into $OLL/models: qwen2.5:{3b,7b,14b,32b,72b}-instruct (Ollama's default Q4_K_M builds).
# Submit one job per judge (MODEL = a tag in overseer/llm/models.py); the 72B needs one 80 GB+ card:
#   sbatch --export=ALL,MODEL=qwen2.5-3b  --time=03:00:00 scripts/create/x5_judge_ollama.sbatch
#   sbatch --export=ALL,MODEL=qwen2.5-72b --time=12:00:00 --constraint="a100_80g|h100|h200|b200" scripts/create/x5_judge_ollama.sbatch
# Output: $REPO/notes/formulation/x5_capability/judge/<tag>_reason.jsonl (resumes, so a requeued or timed-out job
# picks up where it stopped). Partition: gpu or interruptible_gpu, whichever starts first (gpu alone was estimated
# to start in about three weeks on 26 Nov because of the queued RL arrays).
#SBATCH --job-name=x5_judge
#SBATCH --partition=gpu,interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a100|a40|h100|h200|b200
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/users/%u/x5_judge/logs/%j.out
set -euo pipefail
MODEL=${MODEL:?set MODEL, e.g. qwen2.5-3b}
PARALLEL=${PARALLEL:-8}
WORK=/scratch/users/$USER/socialjax_work
OLL=/scratch/users/$USER/ollama
REPO=${REPO:-$WORK/invasion-commons-x5}
LOGDIR=/scratch/users/$USER/x5_judge/logs
mkdir -p "$LOGDIR"
module load python/3.11.6-gcc-13.2.0 2>/dev/null || true
source "$WORK/venv/bin/activate"
cd "$REPO"
export PYTHONPATH=$PWD
export OLLAMA_MODELS=$OLL/models OLLAMA_NUM_PARALLEL=$PARALLEL OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=-1
export OLLAMA_HOST=127.0.0.1:$((20000 + SLURM_JOB_ID % 20000))   # a private port: nodes are shared
OLLAMA_NAME=$(python -c "from overseer.llm.models import BY_TAG; print(BY_TAG['$MODEL']['ollama'])")
echo "$(date) $(hostname) job $SLURM_JOB_ID: $MODEL = $OLLAMA_NAME, parallel $PARALLEL, $OLLAMA_HOST"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

"$OLL/bin/ollama" serve > "$LOGDIR/ollama_$SLURM_JOB_ID.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 60); do curl -sf "http://$OLLAMA_HOST/api/version" >/dev/null && break; sleep 2; done
"$OLL/bin/ollama" --version
# Load the model with the judging context (4,096 = prompt of about 500 tokens + up to 2,048 out) and stop if any of it
# lands on the CPU, which would make the run crawl.
curl -sf "http://$OLLAMA_HOST/api/generate" -d "{\"model\": \"$OLLAMA_NAME\", \"prompt\": \"\", \"keep_alive\": -1, \"options\": {\"num_ctx\": 4096}}" >/dev/null
"$OLL/bin/ollama" ps | tee "$LOGDIR/ps_$SLURM_JOB_ID.txt"
"$OLL/bin/ollama" show "$OLLAMA_NAME" | tee "$LOGDIR/show_${MODEL}.txt"
grep -q "100% GPU" "$LOGDIR/ps_$SLURM_JOB_ID.txt" || { echo "model not fully on the GPU on $(hostname); stopping"; exit 1; }

python -m experiments.llm_overseer.judge_bank --model "$MODEL" --mode reason --max-tokens 2048 --num-ctx 4096 \
    --workers "$PARALLEL" --host "http://$OLLAMA_HOST" --out-dir notes/formulation/x5_capability/judge
echo "$(date) done"
