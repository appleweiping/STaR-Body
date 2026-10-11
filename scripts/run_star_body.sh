#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
runtime_paths="$project_root/video_VACE:$project_root/video_VACE/divibody_training:$project_root/video_VACE/vace/models:$project_root"
if [[ -n "${WAN_SOURCE_ROOT:-}" ]]; then runtime_paths="$runtime_paths:$WAN_SOURCE_ROOT"; fi
if [[ -n "${HMR2_SOURCE_ROOT:-}" ]]; then runtime_paths="$runtime_paths:$HMR2_SOURCE_ROOT"; fi
export PYTHONPATH="$runtime_paths${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_root"
exec "${STAR_BODY_PYTHON:-python}" "$@"
