# Player-only broadcast detection cleanup

This document matches the **Player-Only Detection Cleanup** plan: inference-time controls first, optional polygon ROI, post-merge and track dedup, and a path to a **single-class** retrain.

## Standard `yolo predict` flags (Player class index **3**)

| Flag | Recommended default | Notes |
|------|---------------------|--------|
| `classes` | `3` | Player only; drops Ball, Hoop, graphics classes, etc. |
| `conf` | `0.50` | Grid-search `0.50`–`0.55` before going above **~0.60** (check missed players). |
| `iou` | `0.45` | NMS IoU; lower suppresses same-frame duplicate boxes more aggressively. |
| `max_det` | `30` | Safety cap on crowded frames. |

**CLI (Ultralytics):**

```bash
yolo predict model=path/to/best.pt source=path/to/clip.mov classes=3 conf=0.5 iou=0.45 max_det=30 save=True
```

**Repo helper (same defaults as `src/config.py`):**

```bash
python -m src.predict_clean --model path/to/best.pt --source path/to/clip.mov --save
```

`predict_clean` uses **`model.predict()`** per frame: boxes can **jitter** frame-to-frame; this script only applies **Player** filters and **`show_labels=False` / `show_conf=False`** so saved runs have **clean rectangles** (no `Player 0.84` text).

For **smoother** stakeholder video, use ByteTrack instead:

```bash
python -m src.predict_track --model path/to/best.pt --source path/to/clip.mov --out runs/tracked_demo.mp4
```

Optional extra smoothing: set **`PLAYER_BOX_EMA_ALPHA`** in `src/config.py` to e.g. **`0.4`**, or pass **`--ema-alpha 0.4`** (values below **0.2** are clamped — visible lag). **`1.0`** disables EMA and uses Ultralytics `plot()` only.

| Script | Temporal model | Typical use |
|--------|----------------|-------------|
| `predict_clean` | None (independent frames) | Quick checks, flags A/B |
| `predict_track` | ByteTrack + optional EMA | Exported demo MP4 |

Validate on your own `clip.mov` (not checked into this repo): scan for double boxes, missed defenders, and scoreboard false positives while tuning `(conf, iou)`.

## Spatial pipeline (`infer_spatial`)

[`src/player_tracker.py`](../src/player_tracker.py) applies, **on a copy of the frame** only (court model still sees the full image):

1. **HUD zero-mask** — bottom `HUD_MASK_BOTTOM_PCT` of the frame set to black before player `track()`. **Not a crop** (resolution unchanged).
2. **`classes=[PLAYER_CLASS_ID]`**, **`conf`**, **`iou`**, **`max_det`** from [`src/config.py`](../src/config.py).
3. Optional **foot-in-polygon** filter when ROI is configured (see below).
4. Optional **greedy Player-only IoU merge** (`PLAYER_GREEDY_MERGE_IOU`).
5. Optional **same-frame track dedup** when two ByteTrack IDs overlap (`PLAYER_TRACK_FRAME_DEDUP_IOU`).

Tunables live in `src/config.py`: `PLAYER_PREDICT_CONF`, `PLAYER_PREDICT_IOU`, `PLAYER_PREDICT_MAX_DET`, `HUD_MASK_BOTTOM_PCT`, `COURT_ROI_POLYGON_NORM`, `COURT_ROI_JSON_PATH`, `PLAYER_GREEDY_MERGE_IOU`, `PLAYER_TRACK_FRAME_DEDUP_IOU`, `PLAYER_BOX_LINE_WIDTH`, `PLAYER_BOX_EMA_ALPHA`, `PLAYER_BOX_EMA_STALE_FRAMES`.

## Court ROI JSON schema

Optional file path: `COURT_ROI_JSON_PATH` in `src/config.py`. If the file exists and contains at least **three** vertices, it overrides `COURT_ROI_POLYGON_NORM`.

```json
{
  "polygon_norm": [
    [0.05, 0.1],
    [0.95, 0.1],
    [0.92, 0.75],
    [0.08, 0.75]
  ]
}
```

- Coordinates are **normalized** image space: `x` in `[0, 1]` left→right, `y` in `[0, 1]` top→bottom.
- Vertices in order (e.g. clockwise). The filter uses the bbox **bottom-center** (foot proxy) with `cv2.pointPolygonTest`.
- Empty / missing polygon disables ROI filtering.

## Track ID behavior (known limitation)

If ROI filtering **drops** a player while their foot is outside the polygon (sideline, timeout, etc.), ByteTrack may assign a **new** `track_id` when they re-enter. **Expected** for session-local demos and CSV exports. Stable jersey-level identity needs Re-ID or “hide off-polygon but keep internal tracks” — out of scope unless required.

## Same-frame vs temporal duplicate boxes

- **Same-frame duplicates:** usually **NMS** (`iou`) + **`conf`** (+ greedy merge). Overtuning `iou` can merge **legitimate** close defenders; diagnose with static frames first.
- **Frame-to-frame flicker:** **ByteTrack** (`persist=True`; reset on scene cut per `infer_spatial`). Do not treat as only an NMS problem.

## Player-only retrain (`nc: 1`)

Export filtered labels and train a **single-class** head so the model never allocates capacity to scoreboard classes:

```bash
python -m src.export_player_only_dataset \
  --src-yaml "Basketball Players/dataset.yaml" \
  --out-dir "PlayerOnlyDataset" \
  --symlink

python -m src.train_player --data "PlayerOnlyDataset/dataset.yaml" --epochs 50 --batch 16 --device auto \
  --project spatialTracker --name playerWeightsOneClass
```

Use `--symlink` (default) to avoid copying large image folders; outputs new `labels` and `dataset.yaml` with `nc: 1`, `names: ['Player']`.

## Failure modes

| Symptom | Mitigation |
|--------|------------|
| Double box on one player | Lower `PLAYER_PREDICT_IOU`, raise `conf`, enable merge / track dedup. |
| Legitimate defenders merged | Raise merge IoU or disable greedy merge. |
| Scoreboard / HUD players | Increase `HUD_MASK_BOTTOM_PCT`; tune ROI polygon. |
| Off-court staff boxed | Tighten ROI polygon; optional higher `conf`. |
| `track_id` changes after timeout | Document limitation (above). |
| Zoom / replay / other network | ROI presets per layout; no free lunch without shot-type or registration. |
