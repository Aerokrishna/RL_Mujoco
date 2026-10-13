#!/usr/bin/env bash
# Download the trained models used by the benchmark (GitHub release `models-v1`) into their run folders:
#   projects/forge/runs/jitter_but_works                                          (FORGE)
#   projects/compliant_RL/runs/compliant_peg_s2g_20261012-122716_s0_s2g_bc        (spiral RL warm start)
# TacDiffusion's models come with the `diffusion/TacDiffusion` submodule (git submodule update --init).
#
# Usage (from anywhere):  bash mujoco_rl_bed/scripts/fetch_models.sh        (FORCE=1 to download again)
set -euo pipefail

BED="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"         # .../mujoco_rl_bed
URL="https://github.com/Aerokrishna/RL_Mujoco/releases/download/models-v1"
declare -A SHA=(
  [forge_jitter_but_works.tar.gz]=589ab4ece23ab5476fc7fe4a2bca57213525e6e85170908d948e51a187d8413c
  [spiral_rl_s2g_bc.tar.gz]=2a3a19bce38902755a1a70cf64ebc4678aa997091cf2547d8c3e1c3fd4902153
)
declare -A DEST=(
  [forge_jitter_but_works.tar.gz]=projects/forge/runs/jitter_but_works
  [spiral_rl_s2g_bc.tar.gz]=projects/compliant_RL/runs/compliant_peg_s2g_20261012-122716_s0_s2g_bc
)

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
for f in "${!SHA[@]}"; do
  if [[ -f "$BED/${DEST[$f]}/final_model.zip" && "${FORCE:-0}" != 1 ]]; then
    echo "ok       ${DEST[$f]} (already there)"
    continue
  fi
  echo "download $f"
  curl -fL --retry 3 -o "$tmp/$f" "$URL/$f"
  echo "${SHA[$f]}  $tmp/$f" | sha256sum -c --quiet - || { echo "checksum mismatch: $f" >&2; exit 1; }
  tar xzf "$tmp/$f" -C "$BED"
  echo "ok       ${DEST[$f]}"
done
