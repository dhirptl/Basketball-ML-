"""Constants and court geometry for the spatial tracker pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# --- Repo paths (override via CLI) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAYER_DATA_YAML = REPO_ROOT / "Basketball Players" / "dataset.yaml"
DEFAULT_SPATIAL_PROJECT = REPO_ROOT / "spatialTracker"

# --- NBA court (feet): fixed physical frame ---
# Origin (0,0) = home baseline, left sideline corner.
# x runs 0..94 along long axis (to far baseline). y runs 0..50 (left sideline to right sideline).
COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0

# World (court plane) coordinates for 8 keypoints (strict index order).
# Indices 0-5: corners + center T; 6-7: near-basket FT lane corners (approximate NBA geometry).
_C = COURT_LENGTH_FT / 2.0  # 47 center line
# Free throw line ~19 ft from baseline; lane 16 ft wide centered on y=25 -> y 17..33
_FT_X = 19.0
_FT_Y_L = 17.0
_FT_Y_R = 33.0

COURT_WORLD_POINTS_FT = np.array(
    [
        [0.0, 0.0],  # 0 near-left (home baseline, left sideline)
        [COURT_LENGTH_FT, 0.0],  # 1 far-left
        [_C, 0.0],  # 2 near center T (bottom sideline in broadcast = y=0)
        [_C, COURT_WIDTH_FT],  # 3 far center T (top sideline y=50)
        [0.0, COURT_WIDTH_FT],  # 4 near-right
        [COURT_LENGTH_FT, COURT_WIDTH_FT],  # 5 far-right
        [_FT_X, _FT_Y_L],  # 6 near left FT lane corner
        [_FT_X, _FT_Y_R],  # 7 near right FT lane corner
    ],
    dtype=np.float64,
)

# Priority when selecting points for homography (lower = used first in greedy subset).
KEYPOINT_SOLVE_PRIORITY = np.array([0, 4, 1, 5, 6, 7, 2, 3], dtype=np.int32)

# --- Detection / homography ---
NUM_COURT_KEYPOINTS = 8
KPT_CONF_THRESHOLD = 0.5
MIN_KEYPOINT_SPREAD_PX = 120.0
MAX_HOMOGRAPHY_REPROJ_ERROR = 15.0
HOMOGRAPHY_CONF_SIGMA_PX = 8.0

# RANSAC for homography
HOMOGRAPHY_RANSAC_REPROJ_THRESHOLD = 5.0

# --- Temporal / state ---
ANCHOR_EMA_ALPHA = 0.8  # weight on previous smoothed anchor (higher = smoother)
MAX_HOMOGRAPHY_LOST_FRAMES = 15  # TTL frames without valid registration

# Scene cut: mean absolute grayscale diff (0-255 scale), per-pixel averaged over frame
SCENE_CUT_MEAN_ABS_DIFF = 18.0

# --- Player detection ---
PLAYER_CLASS_NAME = "Player"
# Roboflow Basketball Players v25: Player is index 3
PLAYER_CLASS_ID = 3

# --- Tracking ---
TRACKER_CFG = "bytetrack.yaml"

# --- Inference UX ---
DEBUG_OVERLAY = False
CANONICALIZE_COURT_DIRECTION = False

# --- Training defaults ---
TRAIN_IMGSZ = 640
TRAIN_EPOCHS_PLAYER = 50
TRAIN_EPOCHS_COURT = 100
TRAIN_BATCH = 16
