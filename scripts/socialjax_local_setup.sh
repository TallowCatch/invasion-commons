#!/usr/bin/env bash
# Local (CPU) SocialJax for the overseer wrapper and scripted bots: overseer/socialjax/.
# Clones SocialJax into external/SocialJax (git-ignored) and makes .venv-socialjax with CPU JAX at the versions
# SocialJax pins (jax 0.6.2, flax 0.10.7). Training runs on CREATE instead (scripts/create/setup_socialjax.sh).
# Usage, from the repository root:  bash scripts/socialjax_local_setup.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -d external/SocialJax ] || git clone https://github.com/cooperativex/SocialJax.git external/SocialJax
python3 -m venv .venv-socialjax
.venv-socialjax/bin/pip install -q "jax==0.6.2" "jaxlib==0.6.2" "flax==0.10.7" "chex==0.1.84" "numpy==1.26.4" \
  "gymnax==0.0.8" pillow pytest
SITE="$(.venv-socialjax/bin/python -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n%s\n' "$ROOT" "$ROOT/external/SocialJax" > "$SITE/invasion_commons_socialjax.pth"
echo "SocialJax commit: $(git -C external/SocialJax rev-parse --short HEAD)"
.venv-socialjax/bin/python -c "import jax, socialjax; print('jax', jax.__version__, jax.devices()); socialjax.make('harvest_common_open')"
