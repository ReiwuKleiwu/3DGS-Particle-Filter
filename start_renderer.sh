#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="${BACKEND:-gsplat}"
PORT="${PORT:-8000}"
SPLAT_PATH="${SPLAT_PATH:-$PROJECT_ROOT/splat.ply}"

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
  -p "$PORT:8000" \
  -v "$SPLAT_PATH:/workspace/splat.ply:ro" \
  "$IMAGE_NAME"

echo "Renderer container started."
echo "  container: $CONTAINER_NAME"
echo "  image:     $IMAGE_NAME"
echo "  backend:   $BACKEND"
echo "  nvidia:    $DRIVER_CAPABILITIES"
echo "  splat:     $SPLAT_PATH"
echo "  url:       http://127.0.0.1:$PORT/health"
echo
echo "Useful commands:"
echo "  docker logs -f $CONTAINER_NAME"
echo "  docker rm -f $CONTAINER_NAME"
