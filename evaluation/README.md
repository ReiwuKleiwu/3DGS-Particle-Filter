# Evaluation Tooling

Dieser Ordner ist der eigenständige Einstiegspunkt für die finalen Replay-Experimente der Bachelorarbeit. Er ist so aufgebaut, dass er nicht von `core/replay_tuning` abhängt.

Die Skripte verwenden weiterhin die zentrale Lokalisierungs- und Renderer-Logik aus `core/particle_filter`, `core/config` und `core/rendering`.

## Inhalt

```text
evaluation/
  record_replay_dataset.py      # Replays mit Kamera, Odometrie und Referenzpose aufzeichnen
  generate_splat_csv.py         # explizite Splat-CSV aus einem Splat-Ordner erzeugen
  run_matrix_experiment.py      # Szenario/Pfad/Splat/Mode/Seed-Matrix ausführen
  analyze_matrix_results.py     # per_run_summary.csv aggregieren und Plots erzeugen
  plot_replay_paths.py          # GT-Pfad gegen PF-Schätzung auf der PGM Map plotten
  models.py                     # Replay-Manifest- und Prior-Modelle
  evaluator.py                  # Offline-Observation-Helfer
  paths.py                      # evaluation/artifacts Pfade
  configs/                      # Template-Dateien
  artifacts/                    # Datasets und Results
```

## 1. Replay Aufzeichnen

Beispiel für einen lokalen Nav2-Run mit TF-Referenzpose:

```bash
python3 evaluation/record_replay_dataset.py \
  --name default_small_house_route_1 \
  --goal-x 5.24 \
  --goal-y -0.333 \
  --goal-yaw -0.00525 \
  --record-rate-hz 2.0 \
  --goal-timeout 600 \
  --reference-pose-source tf \
  --tf-time latest
```

Das Dataset landet unter:

```text
evaluation/artifacts/datasets/<name>/
```

Wichtig ist die `manifest.json`; sie wird später in der Experiment-Matrix referenziert.

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
        manifest: evaluation/artifacts/datasets/default_small_house_route_1/manifest.json
      - path_id: route_2
        manifest: evaluation/artifacts/datasets/default_small_house_route_2/manifest.json
      - path_id: route_3
        manifest: evaluation/artifacts/datasets/default_small_house_route_3/manifest.json
      - path_id: route_4
        manifest: evaluation/artifacts/datasets/default_small_house_route_4/manifest.json
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

Der Runner startet den Renderer pro Splat neu, wenn `restart_renderer: true` in der Matrix steht.

## Local Und Global

`local` ist ein Tracking-Test. Der PF startet am ersten Referenzpose-Frame mit fixem Offset:

```text
dx = 0.40 m
dy = 0.00 m
dyaw = 10 deg
particle_count = 500
```

Dieser Prior ist für alle lokalen Runs gleich, damit die Ergebnisse reproduzierbar bleiben.

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
```

`per_run_summary.csv` ist die Grundlage für die wissenschaftliche Aggregation. Jeder Eintrag entspricht:

```text
scenario × path × splat × mode × seed
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
```

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
frame_stride: 5
```

Vor der Hauptstudie sollte immer ein kleiner Pilot-Run ausgeführt werden.
