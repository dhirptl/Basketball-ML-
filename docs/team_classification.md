# Team colour classification (v3.2)

Warm-up collects **quality-gated** colour samples (on-court foot point, min confidence, min box height), builds **pale vs saturated** centroids, validates separation, then classifies by nearest centroid with voting, hysteresis, and label lock.

## Phases

| Phase | On screen | Behaviour |
|-------|-----------|-----------|
| WARMUP | Grey + countdown | Collect samples only on **quality frames** until `_quality_frames_seen >= WARMUP_FRAMES` |
| CALIBRATING | Grey (1 frame) | Outlier filter, pale/sat or K-Means, **centroid separation check**; may extend warm-up |
| LOCKED | Red / blue | Classify only players whose **foot point** is inside court ROI (when enabled) |

`WARMUP_MAX_RAW_FRAMES`: safety cap; if hit, calibration is **forced** (may use fallback centroids if enabled).

## Court ROI (fan / bench rejection)

Use **normalized** polygon (0–1), same as the spatial player path:

- [`COURT_ROI_POLYGON_NORM`](../src/config.py) — list of `(x_norm, y_norm)` vertices, **clockwise or CCW**, ≥3 points.
- Or [`COURT_ROI_JSON_PATH`](../src/config.py) — JSON `{"polygon_norm": [[x,y], ...]}` in **normalized** coordinates.

`TEAM_COURT_ROI_ENABLE = True` with an empty polygon disables foot gating (all boxes pass).

Resolve helper: [`broadcast_preprocess.resolve_court_roi_polygon_norm()`](../src/broadcast_preprocess.py).

To calibrate from pixel clicks on a `W×H` frame: convert each corner to `(x/W, y/H)` and paste into config or JSON.

## Key config (`src/config.py`)

| Parameter | Default | Role |
|-----------|---------|------|
| `TEAM_SLICE_MODE` | `stacked` | `upper_only` (torso) or `stacked` (torso + waist/shorts) |
| `WARMUP_FRAMES` | 120 | Target **quality** frame count |
| `WARMUP_MIN_PLAYERS` | 8 | Min on-court valid detections to count a quality frame |
| `TEAM_WARMUP_MIN_BOX_CONF` | 0.50 | Min detector confidence for warm-up samples |
| `TEAM_WARMUP_MIN_BOX_HEIGHT` | 80 | Min box height (px) for warm-up samples |
| `TEAM_COURT_ROI_ENABLE` | True | Foot-in-polygon gate when polygon has ≥3 vertices |
| `TEAM_CALIB_IQR_*` | — | Saturation IQR outlier trim when sample count is high |
| `TEAM_FILTER_COURT_*` | — | Drop warm-up samples in orange-brown floor hue band |
| `TEAM_CENTROID_MIN_S_SEP` / `MIN_H_SEP` | 15 / 20 | Reject calibration if centroids too close (extend warm-up) |
| `TEAM_PALE_S_THRESHOLD` | 55 | S below → pale pool |
| `TEAM_CENTROID_MARGIN` | 10 | Min distance gap to append to vote history |
| `TEAM_FLIP_MARGIN` | 0.60 | Hysteresis to flip team |
| `TEAM_LOCK_AFTER_FRAMES` | 10 | Lock team per track ID |
| `TEAM_USE_INSTANT_DISPLAY` | True | Show nearest-centroid label while voting |

## CLI

```bash
python -m src.predict_track --model path/to/best.pt --source "okc vs lal.mp4" --out out.mp4
python -m src.predict_track ... --debug-team --debug-slices
python -m src.predict_track ... --debug-team-crops runs/debug_crops --warmup-frames 120
```

`--warmup-frames` overrides `WARMUP_FRAMES` (quality targets).

## Tuning

| Symptom | Try |
|---------|-----|
| Fans/refs in warm-up | Set `COURT_ROI_POLYGON_NORM` / JSON for this camera; keep `TEAM_WARMUP_MIN_BOX_CONF` ≥ 0.5 |
| Never finishes warm-up | Loosen `WARMUP_MIN_PLAYERS`, lower `TEAM_WARMUP_MIN_BOX_HEIGHT`, or verify ROI is not too tight |
| “centroids too close” loop | Widen ROI, lower `TEAM_CALIB_IQR_*` aggressiveness, or adjust `TEAM_PALE_S_THRESHOLD` |
| White jerseys read too saturated | Try `TEAM_SLICE_MODE = "upper_only"` or tune `TEAM_PALE_S_THRESHOLD` |
| Flicker | Raise `TEAM_CENTROID_MARGIN`, `TEAM_FLIP_MARGIN`, `TEAM_LOCK_AFTER_FRAMES` |

## Limits

Identical jersey colours or weak broadcast colour need clip hints, re-ID, or retrained team heads (future).
