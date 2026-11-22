#!/bin/bash -l
# Locality (chi, zeta, Lambda_w, xi_loc) of trained IPPO agents: notes/socialjax/locality/protocol.md, section 5.
# Loads <run>/params.msgpack from the stage 1 (and any later) runs, plays 32 episodes under each run's own overseer
# settings, and measures every region layout of its game. No training; one GPU for about 10-30 min.
#   sbatch --export=ALL,REPO=$WORK/invasion-commons-loc ~/ippo/measure_locality_ippo.sbatch
#   (optional: RUNS="A_none_seed1,B_random_k2_c1.0_seed1", EPISODES=32)
# Output: /scratch/users/$USER/ippo_overseer/locality/locality_ippo.json (resumes run by run)
#SBATCH --job-name=locality
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/%u/ippo_overseer/logs/locality_%j.out
set -euo pipefail

WORK=/scratch/users/$USER/socialjax_work
module load python/3.11.6-gcc-13.2.0 2>/dev/null || echo "module python/3.11.6 not found; using the venv python"
source "$WORK/venv/bin/activate"
cd "${REPO:-$WORK/invasion-commons-loc}"
export PYTHONPATH=$PWD:$WORK/SocialJax:${PYTHONPATH:-}
ROOT=/scratch/users/$USER/ippo_overseer
DEFAULT_RUNS="A_none_seed1,A_none_seed2,A_none_seed3,B_random_k1_c1.0_seed1,B_random_k2_c1.0_seed1,B_random_k4_c1.0_seed1,B_random_k7_c1.0_seed1,B_random_k2_c2.0_seed1,B_random_k2_c0.5_seed1,D_history_k2_c1.0_seed1,D_location_k2_c1.0_seed1"
RUNS=${RUNS:-$DEFAULT_RUNS}
# keep only runs whose params exist
OK=""
for r in ${RUNS//,/ }; do
  if [ -f "$ROOT/$r/params.msgpack" ]; then OK="$OK${OK:+,}$r"; else echo "skip $r (no params.msgpack)"; fi
done
mkdir -p "$ROOT/locality"
python -c "import jax; print('jax backend:', jax.default_backend())"
python -m experiments.socialjax.measure_locality --params-root "$ROOT" --runs "$OK" \
    --episodes "${EPISODES:-32}" --out "$ROOT/locality/locality_ippo.json"
