# Replay Tuning

This folder contains offline replay and tuning tooling for the particle-filter project.

The immediate purpose is:

- record a deterministic traversal from the running robot or simulator
- save camera frames plus `map -> base_link` poses
- replay that exact dataset offline for parameter tuning

## Recorder

Use the Nav2 recorder to send a `NavigateToPose` goal and capture frames until Nav2 reports success:

```bash
python3 -m core.replay_tuning.record_replay_dataset \
  --name hallway_run_01 \
  --goal-x 2.7 \
  --goal-y -3.45 \
  --goal-yaw 1.57 \
  --notes "Nav2 goal capture"
```

Output lands in:

- `core/replay_tuning/artifacts/datasets/<name>/manifest.json`
- `core/replay_tuning/artifacts/datasets/<name>/poses.csv`
- `core/replay_tuning/artifacts/datasets/<name>/raw_capture.json`
- `core/replay_tuning/artifacts/datasets/<name>/images/*.png`

The manifest is the stable handoff format for later offline evaluation and search.

Search results and generated plots are written under:

- `core/replay_tuning/artifacts/results/`

The package layout is intentionally split between importable code and reproducible artifacts:

- Python modules live directly under `core/replay_tuning/`
- recorded datasets live under `core/replay_tuning/artifacts/datasets/`
- generated search outputs live under `core/replay_tuning/artifacts/results/`
