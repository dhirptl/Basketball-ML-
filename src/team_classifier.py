"""v3.0: warm-up calibration + four-slice crops + nearest-centroid team classification."""

from __future__ import annotations

from collections import defaultdict, deque
from enum import Enum, auto

import cv2
import numpy as np

from . import config


class Phase(Enum):
    WARMUP = auto()
    CALIBRATING = auto()
    LOCKED = auto()


def draw_four_slice_debug_lines(vis_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    box_h = y2 - y1
    if box_h < 4:
        return
    slice_h = box_h / 4.0
    for i in range(1, 4):
        line_y = y1 + int(slice_h * i)
        colour = (0, 255, 0) if i in (1, 2) else (0, 0, 255)
        cv2.line(vis_bgr, (x1, line_y), (x2, line_y), colour, 1)


class TeamClassifier:
    """
    Three-phase adaptive team classifier.

    Phase 1 WARMUP: collect four-slice (H, S) samples — all boxes grey.
    Phase 2 CALIBRATING: one-shot K-Means K=2 on pooled samples.
    Phase 3 LOCKED: nearest-centroid assignment + majority vote per track ID.
    """

    def __init__(self, *, warmup_frames: int | None = None) -> None:
        self.phase = Phase.WARMUP
        self._warmup_frames = int(warmup_frames if warmup_frames is not None else config.WARMUP_FRAMES)
        self._samples: list[np.ndarray] = []
        self._frames_seen = 0
        self._centroids: np.ndarray | None = None
        self._history: dict[int, deque[int]] = defaultdict(
            lambda: deque(maxlen=int(config.TEAM_HISTORY_LEN))
        )
        self._team_cache: dict[int, int] = {}
        self._last_debug: dict[int, dict[str, float | int]] = {}

    def reset(self) -> None:
        self.__init__(warmup_frames=self._warmup_frames)

    @property
    def is_locked(self) -> bool:
        return self.phase == Phase.LOCKED

    @property
    def warmup_frames_remaining(self) -> int:
        return max(0, self._warmup_frames - self._frames_seen)

    @property
    def centroids(self) -> np.ndarray | None:
        return self._centroids

    def get_team(self, track_id: int) -> int:
        if not self.is_locked:
            return -1
        return int(self._team_cache.get(int(track_id), -1))

    def get_colour(self, track_id: int) -> tuple[int, int, int]:
        team = self.get_team(track_id)
        if team == 0:
            return tuple(int(c) for c in config.TEAM_A_COLOUR)
        if team == 1:
            return tuple(int(c) for c in config.TEAM_B_COLOUR)
        return tuple(int(c) for c in config.UNKNOWN_TEAM_COLOUR)

    def get_debug_info(self, track_id: int) -> dict[str, float | int] | None:
        return self._last_debug.get(int(track_id))

    def update(self, frame_bgr: np.ndarray, xyxy: np.ndarray, track_ids: np.ndarray) -> None:
        if len(xyxy) == 0:
            return

        if self.phase == Phase.WARMUP:
            self._collect_samples(frame_bgr, xyxy)
            self._frames_seen += 1
            if self._frames_seen >= self._warmup_frames:
                self.phase = Phase.CALIBRATING

        if self.phase == Phase.CALIBRATING:
            self._run_calibration()
            if self._centroids is not None:
                self.phase = Phase.LOCKED

        if self.phase == Phase.LOCKED:
            self._classify_players(frame_bgr, xyxy, track_ids)

    def _collect_samples(self, frame_bgr: np.ndarray, xyxy: np.ndarray) -> None:
        for box in xyxy:
            hs = self._four_slice_hs(frame_bgr, box)
            if hs is not None:
                self._samples.append(hs)

    def _run_calibration(self) -> None:
        if len(self._samples) < 2:
            self.phase = Phase.WARMUP
            return
        samples = np.stack(self._samples, axis=0).astype(np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            int(config.KMEANS_ITER),
            0.5,
        )
        _compact, _labels, centroids = cv2.kmeans(
            samples,
            2,
            None,
            criteria,
            int(config.KMEANS_ATTEMPTS),
            cv2.KMEANS_PP_CENTERS,
        )
        self._centroids = centroids.astype(np.float32)
        print(
            "[TeamClassifier] Calibrated. "
            f"Centroid A: H={centroids[0][0]:.1f} S={centroids[0][1]:.1f} | "
            f"Centroid B: H={centroids[1][0]:.1f} S={centroids[1][1]:.1f}"
        )

    def _classify_players(
        self, frame_bgr: np.ndarray, xyxy: np.ndarray, track_ids: np.ndarray
    ) -> None:
        if self._centroids is None:
            return
        for box, tid in zip(xyxy, track_ids):
            hs = self._four_slice_hs(frame_bgr, box)
            if hs is None:
                continue
            dist_a = self._hs_distance(hs, self._centroids[0])
            dist_b = self._hs_distance(hs, self._centroids[1])
            label = 0 if dist_a <= dist_b else 1
            tid_i = int(tid)
            self._history[tid_i].append(label)
            voted = self._majority_vote(tid_i)
            self._team_cache[tid_i] = voted
            self._last_debug[tid_i] = {
                "dist_a": round(dist_a, 2),
                "dist_b": round(dist_b, 2),
                "label": label,
                "team": voted,
            }

    def _hs_distance(self, hs: np.ndarray, centroid: np.ndarray) -> float:
        dh = abs(float(hs[0]) - float(centroid[0]))
        dh = min(dh, 179.0 - dh)
        ds = float(hs[1]) - float(centroid[1])
        w = float(config.TEAM_HUE_WEIGHT)
        return float(np.sqrt((w * dh) ** 2 + ds**2))

    def _four_slice_hs(self, frame_bgr: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
        fh, fw = frame_bgr.shape[:2]
        x1c = max(0, min(x1, fw - 1))
        x2c = max(x1c + 1, min(x2, fw))
        y1c = max(0, min(y1, fh - 1))
        y2c = max(y1c + 1, min(y2, fh))
        box_h = y2c - y1c
        box_w = x2c - x1c
        if box_h < int(config.MIN_BOX_HEIGHT) or box_w < int(config.TEAM_MIN_BOX_WIDTH_PX):
            return None
        sh = box_h / 4.0
        s1_top = y1c + int(sh * 1)
        s2_top = y1c + int(sh * 2)
        s3_top = y1c + int(sh * 3)
        slice_1 = frame_bgr[s1_top:s2_top, x1c:x2c]
        slice_2 = frame_bgr[s2_top:s3_top, x1c:x2c]
        min_px = int(config.MIN_SLICE_PIXELS)
        if slice_1.size < min_px or slice_2.size < min_px:
            return None
        combined = np.vstack([slice_1, slice_2])
        hsv = cv2.cvtColor(combined, cv2.COLOR_BGR2HSV)
        mean = cv2.mean(hsv)
        return np.array([mean[0], mean[1]], dtype=np.float32)

    def _majority_vote(self, track_id: int) -> int:
        hist = self._history[track_id]
        if len(hist) == 0:
            return -1
        return int(np.argmax(np.bincount(np.asarray(list(hist), dtype=np.int32))))


if __name__ == "__main__":
    clf = TeamClassifier(warmup_frames=3)
    white_box = np.array([10.0, 40.0, 80.0, 160.0], dtype=np.float64)
    blue_box = np.array([100.0, 40.0, 170.0, 160.0], dtype=np.float64)

    for _ in range(3):
        f = np.zeros((200, 200, 3), np.uint8)
        f[50:150, 15:75] = (240, 240, 240)
        f[50:150, 105:165] = (200, 50, 50)
        clf.update(f, np.array([white_box, blue_box]), np.array([1, 2], dtype=np.int64))

    assert clf.is_locked, f"phase={clf.phase}"
    assert clf.centroids is not None
    assert clf.centroids[0][1] < clf.centroids[1][1] or clf.centroids[1][1] < clf.centroids[0][1]
    clf.reset()
    assert clf.phase == Phase.WARMUP and not clf.is_locked
    print("team_classifier v3 tests passed.")
