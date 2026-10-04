#!/usr/bin/env bash
set -euo pipefail

# Create a small source archive for a cloud pod when the latest local changes
# are not yet pushed to GitHub.

OUTPUT="${OUTPUT:-/tmp/harvest_llm_bridge_cloud_payload.tgz}"
tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='results' \
  --exclude='paper/*/*.aux' \
  --exclude='paper/*/*.log' \
  --exclude='paper/*/*.out' \
  --exclude='paper/*/*.pdf' \
  --exclude='notes/*.aux' \
  --exclude='notes/*.log' \
  --exclude='notes/*.out' \
  --exclude='notes/*.pdf' \
  -czf "${OUTPUT}" .
echo "Saved cloud payload: ${OUTPUT}"
