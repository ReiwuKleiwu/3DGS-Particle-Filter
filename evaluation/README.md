# Evaluation Tooling

Die Skripte verwenden die zentrale Lokalisierungs- und Renderer-Logik aus `core/particle_filter`, `core/config` und `core/rendering`.

## Inhalt

```text
evaluation/
  record_replay_dataset_simulator.py  # Simulator-Replays mit Kamera, Odometrie und Referenzpose aufzeichnen
  generate_splat_csv.py         # explizite Splat-CSV aus einem Splat-Ordner erzeugen
  run_matrix_experiment.py      # Szenario/Pfad/Splat/Mode/Seed-Matrix ausführen
  analyze_matrix_results.py     # per_run_summary.csv aggregieren und Plots erzeugen
  plot_replay_paths.py          # GT-Pfad gegen PF-Schätzung auf der PGM Map plotten
  plot_dataset_path.py          # aufgezeichneten Dataset-Pfad direkt aus manifest.json auf der Map plotten
  plot_particle_snapshots.py    # aufgezeichnete Partikelwolken als PNG-Sequenz plotten
  make_particle_video.py        # PNG-Sequenz zu MP4-Video konvertieren
  plot_style.py                 # gemeinsame farbenblindenfreundliche Plot-Stile
  models.py                     # Replay-Manifest- und Prior-Modelle
  evaluator.py                  # Offline-Observation-Helfer
  paths.py                      # evaluation/artifacts Pfade
  configs/                      # Template-Dateien
  artifacts/                    # Datasets und Results
```

## 1. Replay Aufzeichnen

Beispiel für einen lokalen Nav2-Run mit TF-Referenzpose:

```bash
python3 evaluation/record_replay_dataset_simulator.py \
  --name default_small_house_route_1 \
  --goal-x 5.24 \
  --goal-y -0.333 \
  --goal-yaw -0.00525 \
  --record-rate-hz 2.0 \
  --goal-timeout 600 \
  --reference-pose-source tf \
  --tf-time latest
```

Mehrere Nav2-Waypoints können als zusammenhängender Run aufgezeichnet werden:

```bash
python3 evaluation/record_replay_dataset_simulator.py \
  --name default_small_house_route_multi \
  --waypoint 2.0 -1.0 0.0 \
  --waypoint 4.0 -1.0 1.57 \
  --waypoint 5.24 -0.333 -0.00525 \
  --record-rate-hz 2.0 \
  --goal-timeout 180 \
  --reference-pose-source tf \
  --tf-time latest
```

`--goal-timeout` gilt pro Waypoint. Ohne `--waypoint` bleibt der alte Single-Goal-Aufruf mit `--goal-x`, `--goal-y` und `--goal-yaw` gültig.

Für echte TurtleBots mit ROS-Namespace müssen Topics, Action und TF remapped werden. Beispiel für `robot_1`:

```bash
python3 evaluation/record_replay_dataset_simulator.py \
  --name labor_default \
  --image-topic /robot_1/oakd/rgb/preview/image_raw \
  --camera-info-topic /robot_1/oakd/rgb/preview/camera_info \
  --odom-topic /robot_1/odom \
  --amcl-pose-topic /robot_1/amcl_pose \
  --cmd-vel-topic /robot_1/cmd_vel \
  --navigate-to-pose-action /robot_1/navigate_to_pose \
  --map-frame map \
  --base-frame base_link \
  --reference-pose-source tf \
  --tf-time latest \
  --waypoint 3.2171 -5.6471 0.0000003 \
  --waypoint 1.4609 -4.3213 -0.00005 \
  --waypoint 0.32951 0.12498 0.0 \
  --record-rate-hz 2.0 \
  --goal-timeout 600 \
  --ros-args -r /tf:=/robot_1/tf -r /tf_static:=/robot_1/tf_static
```

Die Waypoints sind `map -> base_link` Posen, nicht OAK-D-Kameraposen. `--ros-args` wird vom Recorder an ROS weitergereicht und nicht als Recorder-Argument interpretiert.

Das Dataset landet unter:

```text
evaluation/artifacts/datasets/<name>/
```

Wichtig ist die `manifest.json`; sie wird später in der Experiment-Matrix referenziert.

Das Skript kann sowohl im Simulator als auch im Labor verwendet werden, solange Topics, Nav2-Action und TF-Remapping passend gesetzt sind.

### Dataset-Pfad Schnell Prüfen

Direkt nach dem Recording kann der aufgezeichnete Pfad auf der Map geplottet werden:

```bash
python3 evaluation/plot_dataset_path.py \
  evaluation/artifacts/datasets/labor_default \
  --map-yaml cps_labor_map.yaml \
  --show-waypoints
```

Das erzeugt standardmäßig:

```text
evaluation/artifacts/datasets/labor_default/dataset_path_overlay.png
```

Mit `--full-map` wird die komplette Map statt eines Pfad-Zooms angezeigt.

## 2. Splat CSV Erzeugen

Empfohlene Splat-Stufen:

```text
1000, 2000, 3000, 5000, 8000, 12000, 18000, 30000
```

Beispiel:

```bash
python3 evaluation/generate_splat_csv.py \
  --splat-dir /home/nick/Downloads/brush-app-x86_64-unknown-linux-gnu/small_house \
  --id-prefix small_house \
  --iterations 1000 2000 3000 5000 8000 12000 18000 30000 \
  --output evaluation/configs/default_small_house_splats.csv
```

Standard-Dateimuster:

```text
{prefix}_{iteration:05d}.ply
```

Damit erwartet das Skript z.B.:

```text
small_house_01000.ply
small_house_30000.ply
```

Falls dein Dateiname anders ist, nutze:

```bash
--filename-pattern "small_house_{iteration:05d}.ply"
```

## 3. Experiment-Matrix Anlegen

Kopiere und bearbeite:

```text
evaluation/configs/experiment_matrix.template.yaml
```

Die Hauptstudie sollte pro Szenario vier Pfade enthalten:

```yaml
scenarios:
  - scenario_id: default_small_house
    splat_csv: evaluation/configs/default_small_house_splats.csv
    paths:
      - path_id: route_1
        manifest: evaluation/artifacts/datasets/default_small_house/default_small_house_route_1/manifest.json
      - path_id: route_2
        manifest: evaluation/artifacts/datasets/default_small_house/default_small_house_route_2/manifest.json
      - path_id: route_3
        manifest: evaluation/artifacts/datasets/default_small_house/default_small_house_route_3/manifest.json
      - path_id: route_4
        manifest: evaluation/artifacts/datasets/default_small_house/default_small_house_route_4/manifest.json
```

## 4. Experiment Ausführen

Pilot-Run mit kleiner Matrix:

```bash
python3 evaluation/run_matrix_experiment.py \
  --matrix evaluation/configs/experiment_matrix.pilot.yaml \
  --run-name pilot_small_house
```

Hauptstudie:

```bash
python3 evaluation/run_matrix_experiment.py \
  --matrix evaluation/configs/experiment_matrix.yaml \
  --run-name small_house_main_study
```

Wenn sich Renderer-Code oder Rendering-Offsets geändert haben:

```bash
python3 evaluation/run_matrix_experiment.py \
  --matrix evaluation/configs/experiment_matrix.yaml \
  --run-name small_house_main_study \
  --build-image
```

Der Runner startet den Renderer bei jedem Splat-Wechsel neu, wenn `restart_renderer: true` in der Matrix steht. Matrizen mit mehreren Splats werden ohne diese Option abgelehnt, damit Splat-Vergleiche nicht versehentlich mit einem alten Renderer-Zustand laufen.

Während des Runs zeigt das Skript eine Zusammenfassung der Studie, einen Gesamtfortschrittsbalken und einen Frame-Fortschrittsbalken für den aktuellen Run. Falls die Terminal-Ausgabe in Logs zu unruhig ist:

```bash
python3 evaluation/run_matrix_experiment.py \
  --matrix evaluation/configs/small_house_default/main_matrix.yaml \
  --run-name small_house_default_main_old \
  --no-progress
```

## Local Und Global

`local` ist ein Tracking-Test. Der PF startet am ersten Referenzpose-Frame mit einem von drei festen Offsets:

```text
particle_count = 500

prior_case 0: dx = 0.20 m, dy = 0.00 m, dyaw = 5 deg
prior_case 1: dx = 0.40 m, dy = 0.00 m, dyaw = 10 deg
prior_case 2: dx = 0.50 m, dy = 0.00 m, dyaw = 15 deg
```

Diese Prior-Fälle sind für alle lokalen Runs gleich, damit die Ergebnisse reproduzierbar bleiben.

`global` ist ein globaler Lokalisierungstest:

```text
particle_count = 2000
Initialisierung über freien Map-Raum
```

Für globale Runs werden Konvergenzmetriken gespeichert.

## Outputs

Pro Run entsteht:

```text
evaluation/artifacts/results/<run_name>/
  per_frame.csv
  per_run_summary.csv
  experiment_metadata.json
  particles/                 # optional, nur bei record_particles.enabled=true
```

`per_frame.csv` ist für Pfadplots und Debugging gedacht. Sie enthält pro Frame u.a.:

```text
truth_x, truth_y, truth_yaw
estimate_x, estimate_y, estimate_yaw
x_error_m, y_error_m
translation_error_m
yaw_error_rad, yaw_error_degrees
combined_pose_error_m
effective_particle_count
resampled
render_and_score_ms
total_frame_ms
total_hz
gpu_memory_used_mb
gpu_memory_total_mb
gpu_memory_free_mb
```

Wenn `record_particles` aktiviert ist, schreibt der Runner pro ausgewähltem Run eine CSV unter `particles/<run_id>.csv`. Jede Zeile beschreibt einen Partikel nach einem vollständigen PF-Step:

```text
frame_index, replay_time_s, particle_index
x, y, yaw, weight
recovery_sample, roughening_sample
```

Für komplette Matrix-Runs sollte Partikelrecording sparsam gefiltert werden, z.B. nur ein globaler Run:

```yaml
experiment:
  gpu_memory_poll_stride: 1
  record_particles:
    enabled: true
    frame_stride: 1
    modes: [global]
    seeds: [42]
    path_ids: [route_3]
    splat_ids: [default_small_house_30000]
```

Aufgezeichnete Partikelwolken können anschließend als PNG-Sequenz geplottet werden:

```bash
python3 evaluation/plot_particle_snapshots.py \
  --input-dir evaluation/artifacts/results/<run_name> \
  --map-yaml map.yaml \
  --run-id small_house_default__route_3__default_small_house_30000__global__seed42
```

Aus der PNG-Sequenz kann anschließend ein MP4 erzeugt werden:

```bash
python3 evaluation/make_particle_video.py \
  --frames-dir evaluation/artifacts/results/<run_name>/particle_plots/small_house_default__route_3__default_small_house_30000__global__seed42 \
  --fps 8
```

`per_run_summary.csv` ist die Grundlage für die wissenschaftliche Aggregation. Jeder Eintrag entspricht:

```text
scenario × path × splat × mode × seed
```

Für lokale Runs kommt zusätzlich der feste `prior_case_index` dazu:

```text
scenario × path × splat × local × prior_case × seed
```

`frame_error_by_condition.csv` wird von `analyze_matrix_results.py` erzeugt und mittelt die per-frame Fehler pro:

```text
scenario × path × mode × prior_case × splat × frame_index
```

## Metriken

Primäre Metriken:

```text
mean_translation_error_m
median_translation_error_m
p95_translation_error_m
mean_yaw_error_degrees
p95_yaw_error_degrees
mean_combined_pose_error_m
p95_combined_pose_error_m
failure_rate
lost_tracking_rate
mean_total_frame_ms
mean_total_hz
mean_gpu_memory_used_mb
max_gpu_memory_used_mb
p95_gpu_memory_used_mb
```

Combined Pose Error:

```text
combined_pose_error_m = translation_error_m + 0.5 * abs(yaw_error_rad)
```

Standardgrenzen:

```text
failure:
  translation_error > 0.5 m oder yaw_error > 20 deg

lost_tracking:
  translation_error > 0.75 m oder yaw_error > 30 deg

converged:
  translation_error < 0.25 m und yaw_error < 10 deg
  für 5 aufeinanderfolgende Frames
```

## Analyse

Nach einem Experiment:

```bash
python3 evaluation/analyze_matrix_results.py \
  --input-dir evaluation/artifacts/results/small_house_main_study
```

Das erzeugt:

```text
summary_by_condition.csv
plots/combined_pose_error_by_splat.png
plots/combined_pose_error_by_splat.pdf
plots/failure_rate_by_splat.png
plots/failure_rate_by_splat.pdf
plots/runtime_by_splat.png
plots/runtime_by_splat.pdf
plots/total_hz_by_splat.png
plots/total_hz_by_splat.pdf
plots/gpu_memory_by_splat.png
plots/gpu_memory_by_splat.pdf
plots/convergence_error_by_frame__*.png
plots/convergence_error_by_frame__*.pdf
```

Lokale Prior-Fälle werden in `summary_by_condition.csv` und in den Plots getrennt gehalten.

## Pfadplot

Beispiel für GT-vs-PF auf der Map:

```bash
python3 evaluation/plot_replay_paths.py \
  --input-dir evaluation/artifacts/results/small_house_main_study \
  --map-yaml map.yaml \
  --seed 1001 \
  --mode local \
  --scenario-id default_small_house \
  --path-id route_1 \
  --show-error-links
```

Das erzeugt PNG und PDF im Ergebnisordner.

## Empfohlenes Hauptdesign

```text
4 Pfade pro Szenario
5 Seeds pro Pfad
8 Splat-Stufen: 1k, 2k, 3k, 5k, 8k, 12k, 18k, 30k
local + global
local: 500 Partikel
global: 2000 Partikel
local prior cases: 3
frame_stride: 5
```

Vor der Hauptstudie sollte immer ein kleiner Pilot-Run ausgeführt werden.
