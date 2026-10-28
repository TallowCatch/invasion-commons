#!/bin/bash -l
# One-off setup of SocialJax on KCL CREATE HPC (28 Oct 2026).
# Run inside an interactive session, not on the login node:
#   srun -p cpu --mem=16G --cpus-per-task=4 --time=01:00:00 --pty /bin/bash -l
#   bash ~/setup_socialjax.sh
# Everything lives in scratch (200 GB, not backed up); home is only 50 GB.
set -euo pipefail

ROOT=/scratch/users/$USER/socialjax_work
mkdir -p "$ROOT"
cd "$ROOT"

module load python/3.11.6-gcc-13.2.0

if [ ! -d SocialJax ]; then
  git clone https://github.com/cooperativex/SocialJax.git
fi
cd SocialJax
echo "SocialJax commit: $(git rev-parse HEAD)" | tee "$ROOT/socialjax_commit.txt"

if [ ! -d "$ROOT/venv" ]; then
  python -m venv "$ROOT/venv"
fi
source "$ROOT/venv/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

# Installs pip packages only; the GPU check is done by the smoke job.
python -c "import jax, flax, socialjax; print('jax', jax.__version__, 'flax', flax.__version__, 'socialjax ok')"
echo "Setup done. Next: sbatch ~/smoke_harvest.sbatch"
