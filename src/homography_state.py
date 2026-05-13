"""Homography quality, anchor EMA smoothing, TTL, and scene-cut detection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config
from .court_registrar import CourtKeypoints


def _convex_hull_area_px(pts: np.ndarray) -> float:
    """pts (N,2) float. Returns 0 if degenerate."""
    if pts.shape[0] < 3:
        return 0.0
    hull = cv2.convexHull(pts.astype(np.float32))
    return float(cv2.contourArea(hull))


def _mean_reprojection_error_px(
    H: np.ndarray, src: np.ndarray, dst: np.ndarray
) -> float:
    """src, dst (N,2). H maps homogeneous src to dst."""
    if src.shape[0] == 0:
        return 1e9
    src_h = np.concatenate([src.astype(np.float64), np.ones((src.shape[0], 1))], axis=1)
    proj = (H @ src_h.T).T
    proj = proj[:, :2] / (proj[:, 2:3] + 1e-9)
    err = np.linalg.norm(proj - dst.astype(np.float64), axis=1)
    return float(np.mean(err))


def _ordering_ok(xy: np.ndarray, vis: np.ndarray) -> bool:
    """Lightweight broadcast near/far and left/right checks in image space."""
    margin = 8.0
    # Near row (0,4) should be below far row (1,5) in image y
    if vis[0] and vis[1] and not (xy[0, 1] > xy[1, 1] + margin):
        return False
    if vis[4] and vis[5] and not (xy[4, 1] > xy[5, 1] + margin):
        return False
    # Left column (0,1) x should be left of right column (4,5)
    if vis[0] and vis[4] and not (xy[0, 0] + margin < xy[4, 0]):
        return False
    if vis[1] and vis[5] and not (xy[1, 0] + margin < xy[5, 0]):
        return False
    return True


def homography_confidence_from_reproj(mean_reproj_px: float) -> float:
    if mean_reproj_px >= 1e8 or not np.isfinite(mean_reproj_px):
        return 0.0
    s = float(config.HOMOGRAPHY_CONF_SIGMA_PX)
    v = float(np.exp(-mean_reproj_px / max(s, 1e-6)))
    return float(max(0.0, min(1.0, v)))


@dataclass
class HomographyFrameResult:
    H_image_to_court: np.ndarray | None  # 3x3 maps (x,y,1)_img -> homogeneous court ft
    homography_conf: float
    num_visible_kpts: int
    scene_cut: bool
    registration_ok: bool  # fresh quality pass this frame
    mean_reproj_px: float


class HomographyStateMachine:
    """Scene cut + TTL + anchor EMA + ranked homography solve."""

    def __init__(self, frame_size: tuple[int, int]) -> None:
        self._w, self._h = frame_size
        self._prev_gray: np.ndarray | None = None
        self._ema_img: np.ndarray | None = None  # (K,2)
        self._ema_valid: np.ndarray | None = None  # (K,) bool
        self._lost_frames = 0
        self._H: np.ndarray | None = None
        self._last_good_reproj: float = 1e9

    def _scene_cut_and_update_ref(self, frame_bgr: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        cut = False
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            mad = float(np.mean(np.abs(gray - self._prev_gray)))
            cut = mad >= float(config.SCENE_CUT_MEAN_ABS_DIFF)
        self._prev_gray = gray.copy()
        return cut

    def _try_solve_from_pixels(
        self, img_pts: np.ndarray, world_pts: np.ndarray
    ) -> tuple[np.ndarray | None, float]:
        if img_pts.shape[0] < 4:
            return None, 1e9
        H, mask = cv2.findHomography(
            img_pts.astype(np.float64),
            world_pts.astype(np.float64),
            method=cv2.RANSAC,
            ransacReprojThreshold=float(config.HOMOGRAPHY_RANSAC_REPROJ_THRESHOLD),
        )
        if H is None:
            return None, 1e9
        reproj = _mean_reprojection_error_px(H, img_pts, world_pts)
        if reproj > float(config.MAX_HOMOGRAPHY_REPROJ_ERROR):
            return None, reproj
        return H.astype(np.float64), reproj

    def _greedy_ranked_solve(
        self, xy: np.ndarray, vis: np.ndarray
    ) -> tuple[np.ndarray | None, float, np.ndarray, np.ndarray]:
        """Returns H, mean_reproj, used_img, used_world."""
        order = list(config.KEYPOINT_SOLVE_PRIORITY)
        used_idx: list[int] = []
        for idx in order:
            if 0 <= idx < len(vis) and vis[idx]:
                used_idx.append(int(idx))
            if len(used_idx) >= 8:
                break
        if len(used_idx) < 4:
            return None, 1e9, np.zeros((0, 2)), np.zeros((0, 2))

        # Try subsets: start from most points down to 4, prefer more points
        for n in range(min(len(used_idx), 8), 3, -1):
            trial = used_idx[:n]
            img = xy[trial]
            world = config.COURT_WORLD_POINTS_FT[trial]
            spread = max(np.max(img[:, 0]) - np.min(img[:, 0]), np.max(img[:, 1]) - np.min(img[:, 1]))
            if spread < float(config.MIN_KEYPOINT_SPREAD_PX):
                continue
            area = _convex_hull_area_px(img)
            if area < float(config.MIN_KEYPOINT_SPREAD_PX) ** 2 * 0.25:
                continue
            if not _ordering_ok(xy, vis):
                continue
            H, reproj = self._try_solve_from_pixels(img, world)
            if H is not None:
                return H, reproj, img, world
        return None, 1e9, np.zeros((0, 2)), np.zeros((0, 2))

    def _update_ema(self, xy: np.ndarray, vis: np.ndarray) -> None:
        k = config.NUM_COURT_KEYPOINTS
        if self._ema_img is None:
            self._ema_img = np.full((k, 2), np.nan, dtype=np.float64)
            self._ema_valid = np.zeros((k,), dtype=bool)
        a = float(config.ANCHOR_EMA_ALPHA)
        for i in range(k):
            if not vis[i] or not np.isfinite(xy[i, 0]):
                continue
            if not self._ema_valid[i]:
                self._ema_img[i] = xy[i].astype(np.float64)
                self._ema_valid[i] = True
            else:
                self._ema_img[i] = a * self._ema_img[i] + (1.0 - a) * xy[i].astype(np.float64)

    def _solve_from_ema(self) -> tuple[np.ndarray | None, float]:
        assert self._ema_img is not None and self._ema_valid is not None
        idxs = [i for i in range(config.NUM_COURT_KEYPOINTS) if self._ema_valid[i]]
        if len(idxs) < 4:
            return None, 1e9
        img = self._ema_img[idxs]
        world = config.COURT_WORLD_POINTS_FT[idxs]
        H, reproj = self._try_solve_from_pixels(img, world)
        return H, reproj

    def step(self, frame_bgr: np.ndarray, kpts: CourtKeypoints) -> HomographyFrameResult:
        scene_cut = self._scene_cut_and_update_ref(frame_bgr)
        vis = kpts.visible_mask.copy()
        xy = kpts.xy.copy()
        num_vis = int(np.sum(vis))

        registration_ok = False
        mean_reproj = 1e9
        H_out: np.ndarray | None = None

        if num_vis >= 4 and _ordering_ok(xy, vis):
            H_raw, reproj_raw, used_img, used_world = self._greedy_ranked_solve(xy, vis)
            if H_raw is not None:
                registration_ok = True
                mean_reproj = reproj_raw
                self._update_ema(xy, vis)
                H_smooth, reproj_smooth = self._solve_from_ema()
                if H_smooth is not None:
                    H_out = H_smooth
                    mean_reproj = reproj_smooth
                else:
                    H_out = H_raw
                self._lost_frames = 0
                self._H = H_out
                self._last_good_reproj = float(mean_reproj)

        if not registration_ok:
            self._lost_frames += 1
            if scene_cut:
                self._ema_img = None
                self._ema_valid = None
                self._H = None
                self._lost_frames = 9999
            elif self._H is not None and self._lost_frames <= int(config.MAX_HOMOGRAPHY_LOST_FRAMES):
                H_out = self._H
                mean_reproj = float(self._last_good_reproj)
            else:
                H_out = None
                self._H = None

        conf = homography_confidence_from_reproj(mean_reproj) if H_out is not None else 0.0
        return HomographyFrameResult(
            H_image_to_court=H_out,
            homography_conf=conf,
            num_visible_kpts=num_vis,
            scene_cut=scene_cut,
            registration_ok=registration_ok,
            mean_reproj_px=float(mean_reproj),
        )

    def should_reset_tracker(self, frame: HomographyFrameResult) -> bool:
        return bool(frame.scene_cut and not frame.registration_ok)
