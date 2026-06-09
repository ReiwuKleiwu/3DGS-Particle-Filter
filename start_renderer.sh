#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/turtlebot_localization.yaml}"
BACKEND="${BACKEND:-gsplat}"
PORT="${PORT:-8000}"
SPLAT_PATH="${SPLAT_PATH:-$PROJECT_ROOT/splat.ply}"

config_value() {
  local key="$1"
  local fallback="$2"
  python3 - "$CONFIG_PATH" "$key" "$fallback" <<'PY'
from __future__ import annotations

import sys

try:
    import yaml
except Exception:
    print(sys.argv[3])
    raise SystemExit(0)

config_path, key, fallback = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    value = config.get("renderer", {}).get(key, fallback)
except Exception:
    value = fallback
print(value)
PY
}

SPLAT_MAP_X="${SPLAT_MAP_X:-$(config_value splat_map_x 0.034)}"
SPLAT_MAP_Y="${SPLAT_MAP_Y:-$(config_value splat_map_y -0.028)}"
SPLAT_MAP_SCALE="${SPLAT_MAP_SCALE:-$(config_value splat_map_scale 1.0)}"
SPLAT_MAP_SCALE_X="${SPLAT_MAP_SCALE_X:-$(config_value splat_map_scale_x "$SPLAT_MAP_SCALE")}"
SPLAT_MAP_SCALE_Y="${SPLAT_MAP_SCALE_Y:-$(config_value splat_map_scale_y "$SPLAT_MAP_SCALE")}"
SPLAT_MAP_YAW_DEGREES="${SPLAT_MAP_YAW_DEGREES:-$(config_value splat_map_yaw_degrees 0.0)}"

case "$BACKEND" in
  gsplat)
    DOCKERFILE_PATH="$PROJECT_ROOT/core/renderer_backends/gsplat/Dockerfile"
    IMAGE_NAME="${IMAGE_NAME:-3dgsnav-renderer-gsplat}"
    CONTAINER_NAME="${CONTAINER_NAME:-3dgsnav-renderer-gsplat}"
    DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
    ;;
  vkdiff)
    DOCKERFILE_PATH="$PROJECT_ROOT/core/renderer_backends/vkdiff/Dockerfile"
    IMAGE_NAME="${IMAGE_NAME:-3dgsnav-renderer-vkdiff}"
    CONTAINER_NAME="${CONTAINER_NAME:-3dgsnav-renderer-vkdiff}"
    DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-graphics,compute,utility}"
    ;;
  *)
    echo "Unsupported renderer backend: $BACKEND" >&2
    echo "Supported backends: gsplat, vkdiff" >&2
    exit 1
    ;;
esac

if [[ ! -f "$SPLAT_PATH" ]]; then
  echo "Splat file not found: $SPLAT_PATH" >&2
  exit 1
fi

if [[ "${BUILD_IMAGE:-0}" == "1" ]]; then
  docker build -f "$DOCKERFILE_PATH" -t "$IMAGE_NAME" "$PROJECT_ROOT"
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

docker run -d \
  --gpus all \
  --name "$CONTAINER_NAME" \
  -e "RENDERER_BACKEND=$BACKEND" \
  -e "NVIDIA_DRIVER_CAPABILITIES=$DRIVER_CAPABILITIES" \
  -e "SPLAT_MAP_X=$SPLAT_MAP_X" \
  -e "SPLAT_MAP_Y=$SPLAT_MAP_Y" \
  -e "SPLAT_MAP_SCALE=$SPLAT_MAP_SCALE" \
  -e "SPLAT_MAP_SCALE_X=$SPLAT_MAP_SCALE_X" \
  -e "SPLAT_MAP_SCALE_Y=$SPLAT_MAP_SCALE_Y" \
  -e "SPLAT_MAP_YAW_DEGREES=$SPLAT_MAP_YAW_DEGREES" \
  -p "$PORT:8000" \
  -v "$SPLAT_PATH:/workspace/splat.ply:ro" \
  "$IMAGE_NAME"

echo "Renderer container started."
echo "  container: $CONTAINER_NAME"
echo "  image:     $IMAGE_NAME"
echo "  backend:   $BACKEND"
echo "  nvidia:    $DRIVER_CAPABILITIES"
echo "  splat:     $SPLAT_PATH"
echo "  map align: x=$SPLAT_MAP_X y=$SPLAT_MAP_Y scale=$SPLAT_MAP_SCALE scale_x=$SPLAT_MAP_SCALE_X scale_y=$SPLAT_MAP_SCALE_Y yaw_deg=$SPLAT_MAP_YAW_DEGREES"
echo "  url:       http://127.0.0.1:$PORT/health"
echo
echo "Useful commands:"
echo "  docker logs -f $CONTAINER_NAME"
echo "  docker rm -f $CONTAINER_NAME"
