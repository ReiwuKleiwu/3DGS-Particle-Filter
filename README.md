# 3DGSNav

3DGSNav is a ROS-based localization system that combines a particle filter with a renderer-backed measurement model and a browser-based visualization frontend.

The project is split into two first-party areas:

- `core/`: backend code for localization, rendering, and replay tuning
- `frontend/`: browser UI and the lightweight frontend server

## Architecture

The running system has three processes:

1. Renderer service
   - runs in Docker
   - serves scoring and rendering endpoints on `http://127.0.0.1:8000`
   - supports `gsplat` and `vkdiff`

2. Frontend server
   - serves the browser UI on `http://127.0.0.1:8090`
   - stores the latest visualization snapshot
   - exposes control endpoints used by the localization loop

3. Localization backend
   - runs as `python3 -m core.main`
   - reads ROS topics and TF
   - queries the renderer service
   - publishes snapshots and polls runtime control commands

## Prerequisites

You need:

- Python 3.11+
- a ROS 2 environment with the runtime packages used by this project
  - `rclpy`
  - `tf2_ros`
  - `sensor_msgs`
  - `nav_msgs`
  - `geometry_msgs`
  - `nav2_msgs`
  - related ROS message packages
- Docker
- an NVIDIA GPU runtime for the renderer containers
- a splat file at `./splat.ply`

Important: the Python package metadata in [pyproject.toml](/home/nick/PycharmProjects/3DGSNav/pyproject.toml) does not install ROS itself. ROS remains an external system dependency.

## Python Dependencies

The repo now uses a single root `pyproject.toml` with optional dependency groups.

Install the renderer-related Python dependencies:

```bash
pip install .[rendering]
```

Install replay-tuning dependencies:

```bash
pip install .[tuning]
```

The main localization backend also needs the ROS environment to be sourced before launch.

## Configuration

Runtime configuration lives in [turtlebot_localization.yaml](/home/nick/PycharmProjects/3DGSNav/turtlebot_localization.yaml).

Important defaults:

- renderer URL: `http://127.0.0.1:8000`
- frontend publish URL: `http://127.0.0.1:8090/api/publish-latest`
- frontend control poll URL: `http://127.0.0.1:8090/api/reset-particle-filter/next`
- default renderer backend in config: `vkdiff`

## Running The Full Project

Start the system from the repo root in this order.

### 1. Start the renderer

Default:

```bash
./start_renderer.sh
```

Use `vkdiff` explicitly:

```bash
BACKEND=vkdiff ./start_renderer.sh
```

Force a rebuild:

```bash
BUILD_IMAGE=1 BACKEND=vkdiff ./start_renderer.sh
```

Useful environment variables:

- `BACKEND=gsplat|vkdiff`
- `BUILD_IMAGE=1`
- `SPLAT_PATH=/absolute/path/to/file.ply`
- `PORT=8000`

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### 2. Start the frontend server

```bash
./start_visualization_frontend.sh
```

This serves the UI on:

```text
http://127.0.0.1:8090
```

Health check:

```bash
curl http://127.0.0.1:8090/api/health
```

### 3. Start the localization backend

Make sure your ROS environment is sourced first, then run:

```bash
python3 -m core.main
```

The backend loads [turtlebot_localization.yaml](/home/nick/PycharmProjects/3DGSNav/turtlebot_localization.yaml) by default.

## Common Operations

Start the browser UI:

```text
http://127.0.0.1:8090
```

Follow renderer logs:

```bash
docker logs -f 3dgsnav-renderer-vkdiff
```

Stop the renderer container:

```bash
docker rm -f 3dgsnav-renderer-vkdiff
```

If you started `gsplat`, use `3dgsnav-renderer-gsplat` instead.

## Replay Tuning

Replay-tuning code lives under [core/replay_tuning](/home/nick/PycharmProjects/3DGSNav/core/replay_tuning).

Current structure:

- code: [core/replay_tuning](/home/nick/PycharmProjects/3DGSNav/core/replay_tuning)
- recorded datasets: [core/replay_tuning/artifacts/datasets](/home/nick/PycharmProjects/3DGSNav/core/replay_tuning/artifacts/datasets)
- generated results: [core/replay_tuning/artifacts/results](/home/nick/PycharmProjects/3DGSNav/core/replay_tuning/artifacts/results)

Example recorder command:

```bash
python3 -m core.replay_tuning.record_replay_dataset \
  --name hallway_run_01 \
  --goal-x 2.7 \
  --goal-y -3.45 \
  --goal-yaw 1.57
```

## Repository Layout

```text
core/
  main.py
  particle_filter/
  rendering/
  renderer_backends/
  replay_tuning/
frontend/
  server.py
  index.html
  app.jsx
third_party/
  VkDiffGaussianRasterizer/
start_renderer.sh
start_visualization_frontend.sh
turtlebot_localization.yaml
map.yaml
map.pgm
splat.ply
```
