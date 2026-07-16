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
   - supports both `local` and `global` localization modes

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

Important: the Python package metadata in `pyproject.toml` does not install ROS itself. ROS remains an external system dependency.

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

Runtime configuration lives in `turtlebot_localization.yaml`.

Important defaults:

- renderer URL: `http://127.0.0.1:8000`
- frontend publish URL: `http://127.0.0.1:8090/api/publish-latest`
- frontend control poll URL: `http://127.0.0.1:8090/api/reset-particle-filter/next`
- default renderer backend in config: `vkdiff`
- default initialization mode: `local`

Key config sections:

- `particle_filter`
  - particle count and resampling threshold
- `map`
  - `yaml_path` points to the occupancy-map YAML used by global initialization and the frontend map
- `camera_override`
  - optional focal-length/principal-point adjustment applied to ROS `camera_info`
  - useful when an OAK-D preview stream is cropped/resized and its effective FOV does not match the published intrinsics
- `initial_pose_prior`
  - Gaussian prior used for local localization and local resets
- `motion_noise`
  - noise applied during odometry prediction
- `measurement`
  - renderer-scoring metric and measurement temperature
- `runtime`
  - loop timing, random seed, and stationary-update suspension
- `initialization`
  - startup mode for the particle filter
  - `mode: local | global`
  - `global_yaw_uniform: true | false`
- `recovery`
  - adaptive random-particle recovery used by global localization
  - `enabled`
  - `alpha_slow`
  - `alpha_fast`
  - `random_particle_floor_ratio`
  - `random_particle_max_ratio`
- `adaptive_particle_count`
  - optional heuristic particle-count reduction/expansion
  - disabled by default
  - uses particle spread, renderer scores, recovery activity, and hysteresis

Notes:

- `local` mode initializes from `initial_pose_prior`.
- `global` mode initializes from traversable map free space derived from `map.yaml_path` and the map image referenced inside that YAML.
- Recovery is implemented with augmented-MCL style random-particle injection, so the filter can relocalize after losing track.

### Namespaced Robots

For physical TurtleBots using ROS namespaces such as `/robot_1` or `/robot_2`, set the camera, odometry, and AMCL topics in `turtlebot_localization.yaml`, for example:

```yaml
ros:
  image_topic: /robot_1/oakd/rgb/preview/image_raw
  camera_info_topic: /robot_1/oakd/rgb/preview/camera_info
  odometry_topic: /robot_1/odom
  amcl_pose_topic: /robot_1/amcl_pose
  map_frame: map
  base_frame: base_link
```

TF is usually published on namespaced topics while frame IDs remain `map` and `base_link`. Start the backend with TF remaps:

```bash
python3 -m core.main --ros-args \
  -r /tf:=/robot_1/tf \
  -r /tf_static:=/robot_1/tf_static
```

Use `/robot_2/...` and `/robot_2/tf` for the second robot.

### Map Selection

The map image does not need to be named `map.pgm`. Configure the map YAML path:

```yaml
map:
  yaml_path: cps_labor_map.yaml
```

The image path inside `cps_labor_map.yaml` is resolved relative to that YAML file.

### Camera FOV Override

If real camera images appear more zoomed-in than renderer previews, increase the focal-length scale:

```yaml
camera_override:
  fx_scale: 1.35
  fy_scale: 1.35
  cx_offset: 0.0
  cy_offset: 0.0
```

Larger `fx_scale`/`fy_scale` means narrower rendered FOV. Restart `core.main` after changing these values.

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

For the `vkdiff` backend, LPIPS scoring uses pre-exported ONNX models inside the Docker image. The image currently builds LPIPS models for `320x240`, `300x300`, and `320x320`. If the ROS camera stream uses a different image size with `measurement.metric_name: lpips`, add the matching export in `core/renderer_backends/vkdiff/Dockerfile` and rebuild the image.

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

The backend loads `turtlebot_localization.yaml` by default.

## Frontend Workflow

Open the browser UI at:

```text
http://127.0.0.1:8090
```

The filter controls now include a persistent localization-mode toggle:

- `local`
  - uses the configured Gaussian prior
  - supports map-drawn priors from the UI
  - reset performs a local reinitialization
- `global`
  - reinitializes particles across free map space
  - ignores map-drawn priors
  - reset performs a true global relocalization

The map layer panel also provides toggles for occupancy grid, particles, robot markers, GT pose, AMCL pose, covariance, path history, and density heatmap.

Typical workflows:

### Local startup / tracking

1. Leave the mode toggle in `local`.
2. Optionally left-drag on the map to place a manual prior.
3. Apply the prior or use local reset.
4. Let the filter track from the local Gaussian prior.

### Global startup / relocalization

1. Switch the mode toggle to `global`.
2. Press reset.
3. The backend samples particles from map free space and begins global localization.
4. Adaptive recovery stays active while tracking, so the filter can recover from major failures.

## Common Operations

Follow renderer logs:

```bash
docker logs -f 3dgsnav-renderer-vkdiff
```

Stop the renderer container:

```bash
docker rm -f 3dgsnav-renderer-vkdiff
```

If you started `gsplat`, use `3dgsnav-renderer-gsplat` instead.

## Evaluation And Replay Datasets

Offline replay evaluation lives under `evaluation/`.

Use `evaluation/recording/record_replay_dataset_turtlebot.py` for physical TurtleBot datasets. It records OAK-D images, camera info, odometry history, command velocity, Nav2 feedback, and an AMCL or TF reference pose. In the lab we do not have true ground truth, so AMCL is the practical reference pose for the CPS datasets.

CPS lab matrices live under `evaluation/configs/cps_labor_default/`. The pilot uses the 30k splat, while `main_matrix.yaml` evaluates the standard main-study splat stages 1k, 3k, 8k, 18k, and 30k with local and global modes. See [evaluation/README.md](evaluation/README.md) for the full recorder, evaluation, plotting, and video commands.

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
cps_labor_map.yaml
cps_labor_map.pgm
splat.ply
```
