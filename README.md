# Dual-Model Broadcast Spatial Tracker

Pipeline: **court registration** (YOLO11 pose, 8 floor keypoints) plus **player detection + ByteTrack** (YOLO11 bbox), then **homography** from image feet to NBA court coordinates (feet) with TTL, scene-cut handling, anchor EMA, and quality gates.

## Setup

```bash
cd "/Users/dhirpatel/basketball ML"
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run modules from the repo root so `src` is importable:

```bash
python -m src.train_player --help
python -m src.sample_pose_frames --help
python -m src.train_court_pose --help
python -m src.validate_pose_dataset --help
python -m src.infer_spatial --help
```

## Datasets

- **Player detection (bbox):** [`Basketball Players/dataset.yaml`](Basketball Players/dataset.yaml) — Ultralytics-friendly paths relative to the `Basketball Players` folder (`train/images`, `valid/images`). Class **`Player`** is index **3** in this export.
- **Court pose:** use [`court_pose_template/`](court_pose_template/) as a starting layout (`data.yaml` expects `images/` and `labels/` beside it). Annotate **one class** `Court` with **`kpt_shape: [8, 3]`**.

### Keypoint index contract (strict order)

Annotator view assumes **benches at bottom** (sidelines roughly horizontal): **near** = bottom of frame, **far** = top.

| Index | Name |
|-------|------|
| 0 | Near-Left corner (bottom sideline + left baseline) |
| 1 | Far-Left corner (top sideline + left baseline) |
| 2 | Near center T (bottom sideline + half-court line) |
| 3 | Far center T (top sideline + half-court line) |
| 4 | Near-Right corner (bottom sideline + right baseline) |
| 5 | Far-Right corner (top sideline + right baseline) |
| 6 | Near left free-throw lane corner (near-basket end only) |
| 7 | Near right free-throw lane corner (near-basket end only) |

**Indices 6–7:** only label when the **near** basket’s free-throw lane corners are visible; otherwise set **v = 0** (do not guess). Optional future extension: add far-end FT corners as two more keypoints (10-point schema).

### Coordinate frame (fixed physical court)

- **Origin (0, 0):** home baseline, **left sideline** corner (see `src/config.py` `COURT_WORLD_POINTS_FT` for exact mapping to indices 0–7).
- **x:** 0 → 94 ft along the **long** axis (baseline to opposite baseline).
- **y:** 0 → 50 ft along the **short** axis (left sideline → right sideline).
- **Possession / offense** is **not** encoded in this layer; flip or annotate direction in downstream analytics.
- **`CANONICALIZE_COURT_DIRECTION`** in `src/config.py`: optional experimental flip of exported `(x, y)` (default `False`).

**Bridging frames:** annotators think in **camera-relative near/far**; CSV uses the **fixed court** frame above. When in doubt, align corners 0–5 to `COURT_WORLD_POINTS_FT` mentally before labeling 6–7.

```
  far sideline (top of TV)     y = 50
  +----------------------------------------+
  | 1 far-left        T 3         5 far-R  |
  |                    *                   |
  |              center line x=47         |
  | 0 near-left       T 2         4 near-R|
  +----------------------------------------+
  near sideline (bottom of TV)  y = 0
  x=0 (home baseline) .............. x=94 (far baseline)
```

### `homography_conf` in CSV

Per plan: `mean_reproj_px` = mean reprojection error in **pixels** over keypoints used in the accepted homography; then

`homography_conf = clamp( exp(-mean_reproj_px / HOMOGRAPHY_CONF_SIGMA_PX), 0, 1 )`

Constants in [`src/config.py`](src/config.py). If no valid homography for the frame, **`homography_conf = 0.0`**.

## Training

`Basketball Players/dataset.yaml` intentionally **omits** a top-level `path:` key. In Ultralytics 8.4, `path: .` is resolved as the **process current working directory**, not the folder that contains the YAML, which breaks `train` / `val` paths. With `path` omitted, the dataset root defaults to **this YAML file’s directory** (`Basketball Players/`).

**Player model (async-friendly):**

```bash
python -m src.train_player \
  --data "Basketball Players/dataset.yaml" \
  --epochs 50 --batch 16 --device auto \
  --project spatialTracker --name playerWeights
```

Or open [`notebooks/train_player_async.ipynb`](notebooks/train_player_async.ipynb).

**Sample frames for court annotation:**

```bash
python -m src.sample_pose_frames \
  --images-dir "Basketball Players/train/images" \
  --out-dir pose_sample_export \
  --count 500 --seed 42 --manifest pose_sample_export/manifest.csv
```

**Court pose (after labels exist):**

```bash
python -m src.validate_pose_dataset --labels-dir path/to/labels
python -m src.train_court_pose \
  --data path/to/court_pose_dataset/data.yaml \
  --epochs 100 --batch 16 --device auto \
  --project spatialTracker --name courtWeights
```

Weights paths for inference (defaults depend on where Ultralytics wrote runs):

- `spatialTracker/playerWeights/weights/best.pt`
- `spatialTracker/courtWeights/weights/best.pt`

## Inference

```bash
python -m src.infer_spatial \
  --video path/to/gameplay.mp4 \
  --player-weights spatialTracker/playerWeights/weights/best.pt \
  --court-weights spatialTracker/courtWeights/weights/best.pt \
  --out-csv out/spatial.csv \
  --device auto
```

- **`--debug`:** show OpenCV window overlay (also `DEBUG_OVERLAY` in `src/config.py`).
- **`--debug-out path.mp4`:** write overlay video (no UI); safe for benchmarks.

CSV columns: `frame_idx, track_temp_id, x_ft, y_ft, player_conf, homography_conf, num_visible_kpts`.

**Tracker reset:** ByteTrack IDs reset when **scene-cut** fires **and** registration fails that frame (`HomographyStateMachine.should_reset_tracker`). TTL alone does not force ID reset.

## Failure modes (visual symptoms)

| Symptom | Likely cause |
|--------|----------------|
| Mirrored / flipped court | Wrong keypoint order or swapped near-far labels |
| Coordinates explode | Degenerate homography; check spread / reproj gates |
| Stale drift | Long occlusion; TTL serving last `H` — expected until recovery |
| Replay “jump” | Scene cut + registration loss; new ByteTrack IDs |
| Center-zoom gaps | Only 2 T-points visible; need more court lines in frame (Phase 3: center circle keypoints) |
| Jump shots “teleport” | Z-axis / bbox-bottom proxy (Phase 3: ankle keypoints) |
| Edge-of-frame error | Lens distortion (Phase 3: `undistort` after calibration) |
| ID fragmentation | Temp IDs only; Phase 3: Re-ID to stitch across cuts |

## Phase 1–2 limitations & Phase 3 roadmap

See plan: ground-plane + bbox foot proxy, center-zoom point starvation, radial distortion, and tracklet fragmentation — documented for analysts; mitigations are future modules (pose feet, extra court points, calibration + undistort, appearance Re-ID).

## Layout

| Path | Role |
|------|------|
| `src/config.py` | Court geometry, thresholds, class id |
| `src/device_utils.py` | `mps` → `cuda` → `cpu` |
| `src/court_registrar.py` | Court pose inference |
| `src/homography_state.py` | Ranked solve, quality, EMA anchors, TTL, scene cut |
| `src/player_tracker.py` | Player YOLO + ByteTrack |
| `src/spatial_projector.py` | Bbox bottom-center → court ft |
| `src/infer_spatial.py` | CLI orchestration |
| `src/train_player.py` | Train bbox |
| `src/sample_pose_frames.py` | Sample images for labeling |
| `src/train_court_pose.py` | Train court pose |
| `src/validate_pose_dataset.py` | Label QA |

## License

Basketball Players dataset metadata references CC BY 4.0 (see `Basketball Players/README.roboflow.txt`).
