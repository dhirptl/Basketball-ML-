# Team colour classification (v3.0)

Warm-up calibration discovers the two jersey colour clusters for each clip, then classifies every player by nearest centroid. No hardcoded saturation threshold.

## Pipeline

1. ByteTrack + optional foot dedup (`predict_track`)
2. `TeamClassifier.update(frame, xyxy, ids)` on the **original** frame (not HUD-masked)
3. Draw: team 0 → red, team 1 → blue, unknown/warm-up → grey

## Three phases

| Phase | Frames | On screen | What happens |
|-------|--------|-----------|--------------|
| WARMUP | 0 … `WARMUP_FRAMES`-1 | All boxes **grey** + countdown overlay | Collect (H,S) from four-slice crops; **no** team labels |
| CALIBRATING | 1 frame | Grey | K-Means K=2 on all pooled samples; console prints centroids |
| LOCKED | Rest of clip | **Red / blue** | Nearest-centroid + `TEAM_HISTORY_LEN` majority vote |

In `predict_track`, team centroids stay locked for the whole clip (camera cuts do not reset colours). Scene-cut reset applies only in `infer_spatial` (homography pipeline).

## Four-slice geometry

Slices 1+2 (25%–75% of box height) stacked; slices 0 and 3 discarded. See `--debug-slices` in `predict_track`.

## Config (`src/config.py`)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `WARMUP_FRAMES` | 45 | Samples collected before K-Means (~1.5s @ 30fps) |
| `MIN_BOX_HEIGHT` | 40 | Skip tiny / distant boxes |
| `MIN_SLICE_PIXELS` | 20 | Skip empty crops |
| `TEAM_HISTORY_LEN` | 20 | Majority vote window per track ID |
| `KMEANS_ATTEMPTS` | 10 | K-Means restarts at calibration |
| `KMEANS_ITER` | 20 | K-Means iterations |
| `TEAM_HUE_WEIGHT` | 2.0 | Hue emphasis in centroid distance |

## Tuning

| Symptom | Adjust |
|---------|--------|
| Grey too long at start | Lower `WARMUP_FRAMES` (e.g. 20) |
| Wrong centroids (check console H/S) | Raise `WARMUP_FRAMES` (60–90) or `KMEANS_ATTEMPTS` |
| Flip after lock | Raise `TEAM_HISTORY_LEN` |
| New player slow to colour | Lower `TEAM_HISTORY_LEN` |
| Distant players always grey | Lower `MIN_BOX_HEIGHT` |

Hue guide: H≈0/170 red, H≈110 blue, low S white/grey.

## CLI

```bash
python -m src.predict_track --model path/to/best.pt --source "okc vs lal.mp4" --out out.mp4
python -m src.predict_track ... --warmup-frames 60 --debug-slices --debug-team
```

## Limits

- Fewer than two usable samples during warm-up defers calibration (stays in WARMUP).
- Nearly identical jersey colours need retraining with team labels, not colour heuristics.
- Replays / cuts restart warm-up (grey boxes) by design.
