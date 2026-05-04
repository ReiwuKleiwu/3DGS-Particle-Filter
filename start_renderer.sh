#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-3dgsnav-splat-renderer}"
CONTAINER_NAME="${CONTAINER_NAME:-3dgsnav-splat-renderer}"
PORT="${PORT:-8000}"
SPLAT_PATH="${SPLAT_PATH:-$PROJECT_ROOT/splat.ply}"

if [[ ! -f "$SPLAT_PATH" ]]; then
  echo "Splat file not found: $SPLAT_PATH" >&2
  exit 1
fi

if [[ "${BUILD_IMAGE:-0}" == "1" ]]; then
  docker build -f "$PROJECT_ROOT/splat_renderer/Dockerfile" -t "$IMAGE_NAME" "$PROJECT_ROOT"
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

docker run -d \
  --gpus all \
  --name "$CONTAINER_NAME" \
  -p "$PORT:8000" \
  -v "$SPLAT_PATH:/workspace/splat.ply:ro" \
  "$IMAGE_NAME"

echo "Renderer container started."
echo "  container: $CONTAINER_NAME"
echo "  image:     $IMAGE_NAME"
echo "  splat:     $SPLAT_PATH"
echo "  url:       http://127.0.0.1:$PORT/health"
echo
echo "Useful commands:"
echo "  docker logs -f $CONTAINER_NAME"
echo "  docker rm -f $CONTAINER_NAME"
