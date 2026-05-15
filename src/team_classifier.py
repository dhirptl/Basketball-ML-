"""v3.2: quality warm-up, court ROI via foot point, stacked slices, robust calibration."""

from __future__ import annotations

from collections import defaultdict, deque
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

from . import broadcast_preprocess as bp
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


def _circular_mean_hue(hues: np.ndarray) -> float:
    if hues.size == 0:
        return 0.0
    angles = hues.astype(np.float64) * (2.0 * np.pi / 180.0)
    mean_angle = float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))
    if mean_angle < 0:
        mean_angle += 2.0 * np.pi
    return float(mean_angle * 180.0 / (2.0 * np.pi))


def _circular_hue_diff_deg(h0: float, h1: float) -> float:
    dh = abs(float(h0) - float(h1))
    return float(min(dh, 180.0 - dh))


class TeamClassifier:
    def __init__(
        self,
        *,
        warmup_frames: int | None = None,
        debug_crops_dir: Path | None = None,
        max_debug_crops: int = 12,
    ) -> None:
        self.phase = Phase.WARMUP
        self._warmup_frames = int(warmup_frames if warmup_frames is not None else config.WARMUP_FRAMES)
        self._samples: list[np.ndarray] = []
        self._frames_seen = 0
        self._quality_frames_seen = 0
        self._warmup_force_calibrate = False
        self._centroids: np.ndarray | None = None
        self._history: dict[int, deque[int]] = defaultdict(
            lambda: deque(maxlen=int(config.TEAM_HISTORY_LEN))
        )
        self._team_cache: dict[int, int] = {}
        self._instant_label: dict[int, int] = {}
        self._locked_team: dict[int, int] = {}
        self._lock_streak: dict[int, tuple[int, int]] = {}
        self._last_debug: dict[int, dict[str, float | int]] = {}
        self._debug_crops_dir = debug_crops_dir
        self._max_debug_crops = max_debug_crops
        self._debug_crops_saved = 0
        self._calib_log: str = ""
        self._centroid_reject_count = 0
        self._poly_cache_key: tuple[int, int] | None = None
        self._poly_cache_px: np.ndarray | None = None

    def reset(self) -> None:
        self.__init__(
            warmup_frames=self._warmup_frames,
            debug_crops_dir=self._debug_crops_dir,
            max_debug_crops=self._max_debug_crops,
        )

    @property
    def is_locked(self) -> bool:
        return self.phase == Phase.LOCKED

    @property
    def warmup_frames_remaining(self) -> int:
        return max(0, self._warmup_frames - self._quality_frames_seen)

    @property
    def centroids(self) -> np.ndarray | None:
        return self._centroids

    @property
    def calibration_log(self) -> str:
        return self._calib_log

    def _court_polygon_px(self, frame_h: int, frame_w: int) -> np.ndarray | None:
        if not bool(config.TEAM_COURT_ROI_ENABLE):
            return None
        poly_norm = bp.resolve_court_roi_polygon_norm()
        if len(poly_norm) < 3:
            return None
        key = (frame_h, frame_w)
        if self._poly_cache_key != key:
            self._poly_cache_key = key
            self._poly_cache_px = bp.normalized_polygon_to_pixels(poly_norm, frame_w, frame_h)
        return self._poly_cache_px

    @staticmethod
    def _foot_on_court_mask(xyxy: np.ndarray, polygon_px: np.ndarray | None) -> np.ndarray:
        n = len(xyxy)
        if polygon_px is None:
            return np.ones(n, dtype=bool)
        return np.array(
            [bp.foot_inside_polygon(xyxy[i], polygon_px) for i in range(n)],
            dtype=bool,
        )

    def _warmup_quality_mask(
        self,
        xyxy: np.ndarray,
        confidences: np.ndarray,
        polygon_px: np.ndarray | None,
    ) -> np.ndarray:
        h = xyxy[:, 3] - xyxy[:, 1]
        m = confidences.astype(np.float64) >= float(config.TEAM_WARMUP_MIN_BOX_CONF)
        m &= h >= float(config.TEAM_WARMUP_MIN_BOX_HEIGHT)
        on_court = self._foot_on_court_mask(xyxy, polygon_px)
        m &= on_court
        return m

    def _centroids_separated(self, centroids: np.ndarray) -> bool:
        s_diff = abs(float(centroids[0, 1]) - float(centroids[1, 1]))
        h_diff = _circular_hue_diff_deg(float(centroids[0, 0]), float(centroids[1, 0]))
        return (s_diff >= float(config.TEAM_CENTROID_MIN_S_SEP)) or (
            h_diff >= float(config.TEAM_CENTROID_MIN_H_SEP)
        )

    def get_team(self, track_id: int) -> int:
        if not self.is_locked:
            return -1
        tid = int(track_id)
        if tid in self._locked_team:
            return int(self._locked_team[tid])
        if bool(config.TEAM_USE_INSTANT_DISPLAY) and tid in self._instant_label:
            return int(self._instant_label[tid])
        return int(self._team_cache.get(tid, -1))

    def get_colour(self, track_id: int) -> tuple[int, int, int]:
        team = self.get_team(track_id)
        if team == 0:
            return tuple(int(c) for c in config.TEAM_A_COLOUR)
        if team == 1:
            return tuple(int(c) for c in config.TEAM_B_COLOUR)
        return tuple(int(c) for c in config.UNKNOWN_TEAM_COLOUR)

    def get_debug_info(self, track_id: int) -> dict[str, float | int] | None:
        return self._last_debug.get(int(track_id))

    def update(
        self,
        frame_bgr: np.ndarray,
        xyxy: np.ndarray,
        track_ids: np.ndarray,
        confidences: np.ndarray | None = None,
    ) -> None:
        if len(xyxy) == 0:
            return
        fh, fw = frame_bgr.shape[:2]
        polygon_px = self._court_polygon_px(fh, fw)
        n = len(xyxy)
        if confidences is None or len(confidences) != n:
            confidences = np.ones(n, dtype=np.float64)

        if self.phase == Phase.WARMUP:
            self._frames_seen += 1
            if self._frames_seen >= int(config.WARMUP_MAX_RAW_FRAMES):
                if not self._warmup_force_calibrate:
                    print(
                        "[TeamClassifier] WARNING: WARMUP_MAX_RAW_FRAMES exceeded; forcing calibration.",
                        flush=True,
                    )
                self._warmup_force_calibrate = True
                self.phase = Phase.CALIBRATING
            elif self._frames_seen > int(config.WARMUP_SKIP_INITIAL_FRAMES):
                mask = self._warmup_quality_mask(xyxy, confidences, polygon_px)
                if int(np.count_nonzero(mask)) >= int(config.WARMUP_MIN_PLAYERS):
                    self._collect_samples(frame_bgr, xyxy[mask], track_ids[mask])
                    self._quality_frames_seen += 1
                    if self._quality_frames_seen >= self._warmup_frames:
                        self.phase = Phase.CALIBRATING

        if self.phase == Phase.CALIBRATING:
            self._run_calibration()
            if self._centroids is not None:
                self.phase = Phase.LOCKED

        if self.phase == Phase.LOCKED:
            on_court = self._foot_on_court_mask(xyxy, polygon_px)
            if np.any(on_court):
                self._classify_players(frame_bgr, xyxy[on_court], track_ids[on_court])

    def _collect_samples(
        self, frame_bgr: np.ndarray, xyxy: np.ndarray, track_ids: np.ndarray
    ) -> None:
        cap = int(config.WARMUP_MAX_SAMPLES_PER_FRAME)
        collected: list[np.ndarray] = []
        crops: list[tuple[np.ndarray, int]] = []
        for box, tid in zip(xyxy, track_ids):
            crop, hs = self._four_slice_feature(frame_bgr, box)
            if hs is None:
                continue
            collected.append(hs)
            if crop is not None:
                crops.append((crop, int(tid)))
        if cap > 0 and len(collected) > cap:
            idx = np.linspace(0, len(collected) - 1, cap, dtype=int)
            collected = [collected[i] for i in idx]
            crops = [crops[i] for i in idx if i < len(crops)]
        for hs in collected:
            self._samples.append(hs)
        for crop, tid in crops:
            if (
                self._debug_crops_dir is not None
                and self._debug_crops_saved < self._max_debug_crops
            ):
                self._debug_crops_dir.mkdir(parents=True, exist_ok=True)
                path = self._debug_crops_dir / f"warmup_{self._debug_crops_saved:03d}_id{tid}.png"
                cv2.imwrite(str(path), crop)
                self._debug_crops_saved += 1

    def _filter_outlier_samples(self, samples: np.ndarray) -> np.ndarray:
        if len(samples) < 2:
            return samples
        s_col = samples[:, 1]
        h_col = samples[:, 0]
        keep = (s_col >= 5.0) & (s_col <= 250.0)
        filtered = samples[keep]
        if len(filtered) < 2:
            filtered = samples.copy()

        court = (
            (filtered[:, 0] >= float(config.TEAM_FILTER_COURT_HUE_MIN))
            & (filtered[:, 0] <= float(config.TEAM_FILTER_COURT_HUE_MAX))
            & (filtered[:, 1] < float(config.TEAM_FILTER_COURT_S_MAX))
        )
        fc = filtered[~court]
        if len(fc) >= 2:
            filtered = fc

        min_n = int(config.TEAM_CALIB_IQR_MIN_SAMPLES)
        if len(filtered) >= min_n:
            sv = filtered[:, 1].astype(np.float64)
            q1, q3 = np.percentile(
                sv,
                [float(config.TEAM_CALIB_IQR_PCT_LOW), float(config.TEAM_CALIB_IQR_PCT_HIGH)],
            )
            iqr = q3 - q1
            f = float(config.TEAM_CALIB_IQR_FACTOR)
            lo, hi = q1 - f * iqr, q3 + f * iqr
            iqr_ok = (sv >= lo) & (sv <= hi)
            cand = filtered[iqr_ok]
            if len(cand) >= 2:
                filtered = cand

        return filtered if len(filtered) >= 2 else samples

    def _run_calibration(self) -> None:
        force = self._warmup_force_calibrate
        self._warmup_force_calibrate = False

        if len(self._samples) < 2:
            if force and bool(config.TEAM_CALIB_FALLBACK_CENTROIDS):
                self._centroids = np.array([[15.0, 50.0], [95.0, 120.0]], dtype=np.float32)
                self._calib_log = (
                    "[TeamClassifier] Forced calibration: insufficient samples; using fallback centroids."
                )
                print(self._calib_log, flush=True)
                return
            self.phase = Phase.WARMUP
            return

        samples = np.stack(self._samples, axis=0).astype(np.float32)
        samples = self._filter_outlier_samples(samples)
        if len(samples) < 2:
            if force and bool(config.TEAM_CALIB_FALLBACK_CENTROIDS):
                self._centroids = np.array([[15.0, 50.0], [95.0, 120.0]], dtype=np.float32)
                self._calib_log = "[TeamClassifier] Forced calibration: outliers left <2 samples; fallback centroids."
                print(self._calib_log, flush=True)
                return
            self.phase = Phase.WARMUP
            return

        pale_thresh = float(config.TEAM_PALE_S_THRESHOLD)
        pale_mask = samples[:, 1] < pale_thresh
        pale = samples[pale_mask]
        sat = samples[~pale_mask]
        if bool(config.TEAM_WARMUP_BALANCE_SAMPLES):
            n_bal = min(len(pale), len(sat))
            if n_bal >= 3:
                rng = np.random.default_rng(42)
                pale = pale[rng.choice(len(pale), n_bal, replace=False)]
                sat = sat[rng.choice(len(sat), n_bal, replace=False)]

        if len(pale) >= 3 and len(sat) >= 3:
            c_pale = np.mean(pale, axis=0)
            c_sat = np.mean(sat, axis=0)
            mode = "pale_vs_saturated"
        else:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                int(config.KMEANS_ITER),
                0.5,
            )
            _compact, _labels, centers = cv2.kmeans(
                samples,
                2,
                None,
                criteria,
                int(config.KMEANS_ATTEMPTS),
                cv2.KMEANS_PP_CENTERS,
            )
            c_pale, c_sat = centers[0], centers[1]
            mode = "kmeans_fallback"

        centroids = np.stack([c_pale, c_sat], axis=0).astype(np.float32)
        if bool(config.TEAM_AUTO_ANCHOR_PALER_TO_A) and centroids[1, 1] < centroids[0, 1]:
            centroids = centroids[[1, 0]]

        if not force and not self._centroids_separated(centroids):
            self._centroid_reject_count += 1
            every = max(0, int(config.TEAM_CALIB_REJECT_LOG_EVERY))
            if every == 0 or self._centroid_reject_count % max(every, 1) == 0:
                print(
                    "[TeamClassifier] WARNING: centroids too close (S/hue separation); "
                    "extending warm-up. Reject count="
                    f"{self._centroid_reject_count}",
                    flush=True,
                )
            self._samples = []
            self._quality_frames_seen = 0
            self.phase = Phase.WARMUP
            self._centroids = None
            return

        self._centroids = centroids
        pale_is_a = centroids[0, 1] <= centroids[1, 1]
        self._calib_log = (
            f"[TeamClassifier] Calibrated ({mode}). "
            f"Team0/pale: H={centroids[0][0]:.1f} S={centroids[0][1]:.1f} | "
            f"Team1/coloured: H={centroids[1][0]:.1f} S={centroids[1][1]:.1f} | "
            f"pale_samples={len(pale)} sat_samples={len(sat)} "
            f"({'lower-S→Team0' if pale_is_a else 'check anchor'})"
        )
        print(self._calib_log, flush=True)

    def _classify_players(
        self, frame_bgr: np.ndarray, xyxy: np.ndarray, track_ids: np.ndarray
    ) -> None:
        if self._centroids is None:
            return
        margin = float(config.TEAM_CENTROID_MARGIN)
        for box, tid in zip(xyxy, track_ids):
            hs = self._four_slice_feature(frame_bgr, box)[1]
            if hs is None:
                continue
            dist_a = self._hs_distance(hs, self._centroids[0])
            dist_b = self._hs_distance(hs, self._centroids[1])
            label = 0 if dist_a <= dist_b else 1
            confident = abs(dist_a - dist_b) >= margin
            tid_i = int(tid)
            self._instant_label[tid_i] = label

            if confident and tid_i not in self._locked_team:
                self._history[tid_i].append(label)
                voted = self._majority_vote_with_hysteresis(tid_i)
                self._team_cache[tid_i] = voted
                self._update_lock(tid_i, voted)

            self._last_debug[tid_i] = {
                "H": round(float(hs[0]), 1),
                "S": round(float(hs[1]), 1),
                "dist_a": round(dist_a, 2),
                "dist_b": round(dist_b, 2),
                "frame_label": label,
                "confident": int(confident),
                "team": self.get_team(tid_i),
            }

    def _update_lock(self, track_id: int, voted: int) -> None:
        if voted < 0:
            return
        streak_label, streak_len = self._lock_streak.get(track_id, (-1, 0))
        if voted == streak_label:
            streak_len += 1
        else:
            streak_label, streak_len = voted, 1
        self._lock_streak[track_id] = (streak_label, streak_len)
        if streak_len >= int(config.TEAM_LOCK_AFTER_FRAMES):
            self._locked_team[track_id] = voted

    def _majority_vote_with_hysteresis(self, track_id: int) -> int:
        hist = self._history[track_id]
        if len(hist) == 0:
            return -1
        counts = np.bincount(np.asarray(list(hist), dtype=np.int32), minlength=2)
        winner = int(np.argmax(counts))
        current = self._team_cache.get(track_id, -1)
        if current < 0:
            return winner
        need = max(1, int(len(hist) * float(config.TEAM_FLIP_MARGIN)))
        if winner != current and counts[winner] < need:
            return current
        return winner

    def _hs_distance(self, hs: np.ndarray, centroid: np.ndarray) -> float:
        dh = abs(float(hs[0]) - float(centroid[0]))
        dh = min(dh, 179.0 - dh)
        ds = float(hs[1]) - float(centroid[1])
        w = float(config.TEAM_HUE_WEIGHT)
        return float(np.sqrt((w * dh) ** 2 + ds**2))

    def _four_slice_feature(
        self, frame_bgr: np.ndarray, box: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
        fh, fw = frame_bgr.shape[:2]
        x1c = max(0, min(x1, fw - 1))
        x2c = max(x1c + 1, min(x2, fw))
        y1c = max(0, min(y1, fh - 1))
        y2c = max(y1c + 1, min(y2, fh))
        box_h = y2c - y1c
        box_w = x2c - x1c
        if box_h < int(config.MIN_BOX_HEIGHT) or box_w < int(config.TEAM_MIN_BOX_WIDTH_PX):
            return None, None
        sh = box_h / 4.0
        s1_top = y1c + int(sh * 1)
        s2_top = y1c + int(sh * 2)
        s3_top = y1c + int(sh * 3)
        slice_1 = frame_bgr[s1_top:s2_top, x1c:x2c]
        if str(config.TEAM_SLICE_MODE).lower() == "upper_only":
            combined = slice_1
        else:
            slice_2 = frame_bgr[s2_top:s3_top, x1c:x2c]
            min_px = int(config.MIN_SLICE_PIXELS)
            if slice_1.size < min_px or slice_2.size < min_px:
                return None, None
            combined = np.vstack([slice_1, slice_2])
        if combined.size < int(config.MIN_SLICE_PIXELS):
            return None, None
        hsv = cv2.cvtColor(combined, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].astype(np.float32)
        h = hsv[:, :, 0].astype(np.float32)
        median_s = float(np.median(s))
        hue_min = float(config.TEAM_S_MIN_FOR_HUE)
        mask = s >= hue_min
        mean_h = _circular_mean_hue(h[mask]) if np.any(mask) else float(np.median(h))
        return combined, np.array([mean_h, median_s], dtype=np.float32)


if __name__ == "__main__":
    _restore: dict[str, object] = {}
    for _key in (
        "WARMUP_SKIP_INITIAL_FRAMES",
        "WARMUP_MIN_PLAYERS",
        "TEAM_COURT_ROI_ENABLE",
        "WARMUP_FRAMES",
        "TEAM_WARMUP_MIN_BOX_HEIGHT",
        "TEAM_WARMUP_MIN_BOX_CONF",
        "TEAM_CENTROID_MIN_S_SEP",
        "TEAM_CENTROID_MIN_H_SEP",
        "TEAM_CALIB_IQR_MIN_SAMPLES",
    ):
        _restore[_key] = getattr(config, _key)
    config.WARMUP_SKIP_INITIAL_FRAMES = 0
    config.WARMUP_MIN_PLAYERS = 2
    config.TEAM_COURT_ROI_ENABLE = False
    config.WARMUP_FRAMES = 5
    config.TEAM_WARMUP_MIN_BOX_HEIGHT = 40.0
    config.TEAM_WARMUP_MIN_BOX_CONF = 0.1
    config.TEAM_CENTROID_MIN_S_SEP = 1.0
    config.TEAM_CENTROID_MIN_H_SEP = 1.0
    config.TEAM_CALIB_IQR_MIN_SAMPLES = 10_000
    clf = TeamClassifier(warmup_frames=5)
    white_box = np.array([10.0, 40.0, 80.0, 160.0], dtype=np.float64)
    blue_box = np.array([100.0, 40.0, 170.0, 160.0], dtype=np.float64)
    boxes = np.stack([white_box, blue_box] * 3)
    ids = np.arange(6, dtype=np.int64)
    confs = np.ones(6, dtype=np.float64)
    for _ in range(5):
        f = np.zeros((200, 200, 3), np.uint8)
        # Cover full vertical span inside boxes (stacked slices include lower quarter).
        f[40:140, 15:75] = (255, 255, 255)
        f[40:140, 105:165] = (0, 255, 0)
        clf.update(f, boxes, ids, confs)
    assert clf.is_locked, clf.phase
    assert clf.centroids is not None
    assert clf.centroids[0, 1] < clf.centroids[1, 1]
    for _k, _v in _restore.items():
        setattr(config, _k, _v)
    print("team_classifier v3.2 tests passed.")
