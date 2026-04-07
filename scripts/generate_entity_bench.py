import os
import shutil
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import cv2

"""EntityBench generator (final structure)

Design goal (recommended):
  4 tasks × 2 object-types × 2 background-classes × 15 trajectory variants = 240

Tasks (4):
  1) cross: two lines with opposite slopes (identity swap / attention conflict)
  2) occlusion: one static, one passes through (re-appearance / layering failure)
  3) parallel: two parallel lines (sanity check)
  4) attribute_confusion (killer): light-blue vs deep-blue, circle vs ellipse (fine-grained binding)

Diversity dimensions:
  - object_type (2): sphere (no direction) vs structured (cube/prism/arrow)
  - bg_class (2): minimalist (pure colors) vs structured (grid / marble)
  - trajectory variants (15): jitter + non-symmetry

Compatibility note (Wan-Move):
  - Outputs a first-frame image + tracks.npy + visibility.npy + prompt.txt.
  - Extra meta.json is optional and won’t break Wan-Move.

Run:
  python entitybench_generate_final.py

Optional lite config (~180 cases):
  python entitybench_generate_final.py --lite180
  (keeps 15 variants for 3 tasks, but only 5 variants for attribute_confusion)
"""

# ----------------------------
# Defaults
# ----------------------------

DEFAULT_SAVE_ROOT = "data/EntityBench"
NUM_FRAMES = 81
W, H = 832, 480
HORIZON_Y = int(H * 0.35)

TASKS = ["cross", "occlusion", "parallel", "attribute_confusion"]
# Optional extra task for failure mining (not included in TASKS by default)
OPTIONAL_TASKS = ["identity_stress"]

OBJECT_TYPES = ["sphere", "structured"]
BG_CLASSES = ["minimalist", "structured"]

LIGHT_DIRS = ["left", "right", "top"]

# OpenCV uses BGR
BGR = {
    "red": (50, 50, 255),
    "blue": (255, 50, 50),
    "green": (50, 255, 50),
    "yellow": (50, 255, 255),
    "purple": (200, 60, 200),
    "orange": (60, 140, 255),
    "cyan": (255, 255, 50),
    "magenta": (255, 50, 255),
    "light_blue": (255, 200, 120),
    "deep_blue": (180, 70, 20),
}

STYLE_WORDS = ["glowing", "neon", "metallic", "matte", "shiny"]


# ----------------------------
# Seeding / utils
# ----------------------------

def seeded_rng(*parts) -> np.random.Generator:
    s = "|".join(map(str, parts)).encode("utf-8")
    seed = (sum(s) + len(s) * 131) % (2**32 - 1)
    return np.random.default_rng(seed)


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-6)


def get_light_vector(light_dir: str) -> np.ndarray:
    if light_dir == "left":
        return normalize(np.array([-1, -1, 1], dtype=np.float32))
    if light_dir == "right":
        return normalize(np.array([1, -1, 1], dtype=np.float32))
    return normalize(np.array([0, -1, 1], dtype=np.float32))


def pick_light(task: str, obj_type: str, bg_class: str, variant_id: int) -> str:
    key = f"{task}|{obj_type}|{bg_class}|{variant_id}"
    idx = (sum(key.encode("utf-8")) + 17 * len(key)) % len(LIGHT_DIRS)
    return LIGHT_DIRS[idx]


def get_scale(y_px: int) -> float:
    norm = (y_px - HORIZON_Y) / (H - HORIZON_Y)
    return 0.30 + 1.20 * (norm**1.5)


def rot2d(pts_xy: np.ndarray, angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float32)
    return (pts_xy @ R.T).astype(np.float32)


# ----------------------------
# Backgrounds (4 variants)
# ----------------------------

def _apply_global_light(img: np.ndarray, light_dir: str, strength: float = 0.18) -> np.ndarray:
    h, w = img.shape[:2]
    if light_dir == "left":
        g = np.tile(np.linspace(1.0 + strength, 1.0 - strength, w), (h, 1))
    elif light_dir == "right":
        g = np.tile(np.linspace(1.0 - strength, 1.0 + strength, w), (h, 1))
    else:
        g = np.tile(np.linspace(1.0 + strength * 0.5, 1.0 - strength, h)[:, None], (1, w))
    out = img.astype(np.float32) * g[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def bg_pure(color_bgr: Tuple[int, int, int], light_dir: str) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = np.array(color_bgr, dtype=np.uint8)
    # slight horizon band
    cv2.rectangle(
        img,
        (0, HORIZON_Y - 2),
        (W, HORIZON_Y + 2),
        (max(color_bgr[0] - 10, 0), max(color_bgr[1] - 10, 0), max(color_bgr[2] - 10, 0)),
        -1,
    )
    return _apply_global_light(img, light_dir, strength=0.10)


def _apply_aerial_perspective(img: np.ndarray, fog_strength: float = 0.55, blur_sigma: float = 2.2) -> np.ndarray:
    """Add aerial perspective (haze) near horizon to strengthen depth cues."""
    out = img.astype(np.float32)

    # weight: 1 near horizon, 0 near bottom
    y = np.arange(H, dtype=np.float32)
    w = np.zeros((H,), dtype=np.float32)
    if H - HORIZON_Y > 1:
        t = (y - HORIZON_Y) / (H - HORIZON_Y)
        t = np.clip(t, 0, 1)
        # near horizon (t≈0) -> strong haze
        w = (1.0 - t) ** 1.6

    # only apply on floor area
    w[:HORIZON_Y] = 0.0
    w2 = w[:, None, None]  # (H,1,1)

    fog_color = np.array([40, 40, 40], dtype=np.float32)[None, None, :]  # (1,1,3)
    out = out * (1.0 - fog_strength * w2) + fog_color * (fog_strength * w2)

    out = np.clip(out, 0, 255).astype(np.uint8)

    # horizon blur: blend blurred version near horizon
    blurred = cv2.GaussianBlur(out, (0, 0), blur_sigma)
    alpha = (w2 * 0.85).astype(np.float32)
    out = (out.astype(np.float32) * (1 - alpha) + blurred.astype(np.float32) * alpha)

    return np.clip(out, 0, 255).astype(np.uint8)


def bg_grid(light_dir: str, grid_color=(120, 120, 120), base_color=(25, 25, 25)) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:HORIZON_Y] = (15, 15, 15)
    img[HORIZON_Y:] = np.array(base_color, dtype=np.uint8)

    vp = (W // 2, HORIZON_Y)
    for x in range(-W, 2 * W, 80):
        cv2.line(img, (x, H), vp, grid_color, 1, cv2.LINE_AA)

    for i in range(1, 22):
        y = HORIZON_Y + int((i / 22) ** 2 * (H - HORIZON_Y))
        cv2.line(img, (0, y), (W, y), (90, 90, 90), 1, cv2.LINE_AA)

    # micro noise (film grain)
    noise = (np.random.randn(H, W, 1) * 3).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    img = _apply_global_light(img, light_dir, strength=0.16)
    img = _apply_aerial_perspective(img, fog_strength=0.60, blur_sigma=2.6)
    return img


def _marble_texture(h: int, w: int, base_rgb=(220, 215, 210), vein_rgb=(90, 90, 95), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)

    n1 = rng.random((h, w)).astype(np.float32)
    n2 = cv2.resize(rng.random((max(8, h // 4), max(8, w // 4))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    n3 = cv2.resize(rng.random((max(8, h // 16), max(8, w // 16))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    noise = 0.55 * n1 + 0.30 * n2 + 0.15 * n3
    noise = cv2.GaussianBlur(noise, (0, 0), 6)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    freq = 0.055
    amp = 12.0
    phase = noise * 8.0
    map_x = xx + amp * np.sin((yy * freq) + phase)
    map_y = yy + amp * np.cos((xx * freq) + phase)

    warped = cv2.remap(noise, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    veins = np.sin((xx * 0.045) + warped * 12.0)
    veins = np.abs(veins)
    veins = cv2.GaussianBlur(veins, (0, 0), 2)
    veins = (veins - veins.min()) / (veins.max() - veins.min() + 1e-6)

    base = np.array(base_rgb, dtype=np.float32)
    vein = np.array(vein_rgb, dtype=np.float32)
    mask = np.clip((veins - 0.70) / 0.30, 0, 1)

    col = base[None, None, :] * (1 - mask[..., None]) + vein[None, None, :] * mask[..., None]
    shade = 0.92 + 0.16 * warped
    col = col * shade[..., None]

    return np.clip(col, 0, 255).astype(np.uint8)


def _warp_floor_perspective(floor_tex: np.ndarray, horizon_y: int) -> np.ndarray:
    """Warp a top-down floor texture into a perspective floor trapezoid.

    The key is: near area shows larger texels, far area shows denser texels.
    """
    h_floor, w_floor = floor_tex.shape[:2]

    # Source: full texture rectangle
    src = np.array([[0, 0], [w_floor - 1, 0], [w_floor - 1, h_floor - 1], [0, h_floor - 1]], dtype=np.float32)

    # Destination: trapezoid on the final image floor region
    far_w = int(W * 0.55)   # width near horizon
    near_w = int(W * 1.05)  # width near bottom (slightly wider than canvas)
    x_far0 = (W - far_w) // 2
    x_far1 = x_far0 + far_w
    x_near0 = (W - near_w) // 2
    x_near1 = x_near0 + near_w

    dst = np.array(
        [
            [x_far0, horizon_y],
            [x_far1, horizon_y],
            [x_near1, H - 1],
            [x_near0, H - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(floor_tex, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # mask outside the trapezoid (keep black; will be overwritten by wall anyway)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst.astype(np.int32), 255)

    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[mask == 255] = warped[mask == 255]
    return out


def bg_marble(light_dir: str, marble_seed: int) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # wall: keep as 2D (less perceptual impact)
    wall = _marble_texture(HORIZON_Y, W, base_rgb=(60, 60, 65), vein_rgb=(20, 20, 22), seed=marble_seed + 11)
    img[:HORIZON_Y] = wall

    # floor: generate a large top-down texture then warp with perspective
    # Using a larger texture gives richer near-field detail.
    floor_tex = _marble_texture(
        h=max(480, (H - HORIZON_Y) * 2),
        w=max(1024, W * 2),
        base_rgb=(220, 215, 210),
        vein_rgb=(95, 95, 105),
        seed=marble_seed + 29,
    )

    floor_p = _warp_floor_perspective(floor_tex, HORIZON_Y)
    # composite: floor only below horizon
    img[HORIZON_Y:] = np.maximum(img[HORIZON_Y:], floor_p[HORIZON_Y:])

    # subtle seams on the *image space* floor (keeps perspective)
    for x in range(0, W, 104):
        cv2.line(img, (x, HORIZON_Y), (x, H), (175, 175, 180), 1, cv2.LINE_AA)

    img = _apply_global_light(img, light_dir, strength=0.14)
    img = _apply_aerial_perspective(img, fog_strength=0.40, blur_sigma=2.0)
    return img


def choose_background(bg_class: str, task: str, obj_type: str, variant_id: int, case_id: int, light_dir: str):
    """Return (bg_variant_name, bg_image, bg_seed_or_None)."""
    if bg_class == "minimalist":
        if case_id % 2 == 0:
            return "pure_black", bg_pure((0, 0, 0), light_dir), None
        return "pure_white", bg_pure((255, 255, 255), light_dir), None

    if bg_class == "structured":
        if case_id % 2 == 0:
            return "grid", bg_grid(light_dir), None

        # task-aware marble seed (avoid case-id shortcut)
        s = f"{task}|{obj_type}|{variant_id}".encode("utf-8")
        marble_seed = (sum(s) + 9973 * len(s)) % 1_000_000
        return "marble", bg_marble(light_dir, marble_seed), marble_seed

    raise ValueError(f"Unknown bg_class: {bg_class}")


# ----------------------------
# Trajectory variants (15)
# ----------------------------

@dataclass
class Tracks:
    tr1: np.ndarray  # (T,2) in [0,1]
    tr2: np.ndarray  # (T,2) in [0,1]
    meta: Dict


def _smooth_noise(rng: np.random.Generator, t: np.ndarray, amp: float) -> np.ndarray:
    knots = 8
    xs = np.linspace(0, 1, knots)
    ys = rng.normal(0, 1, size=knots).astype(np.float32)
    ys = cv2.GaussianBlur(ys.reshape(-1, 1), (1, 0), 1.2).reshape(-1)
    noise = np.interp(t, xs, ys)
    noise = noise / (np.std(noise) + 1e-6)
    return amp * noise


def _ease_in_out(u: np.ndarray) -> np.ndarray:
    """Smoothstep easing in [0,1]."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def _apply_overlap_event(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    mid: int,
    x_ov: float,
    y_ov: float,
    hold: int,
    ramp: int,
):
    """Make an overlap/meeting event without teleportation.

    Structure:
      ramp-in (ramp frames) -> hold (hold frames) -> ramp-out (ramp frames)

    During hold, both objects share the same coordinates.
    Ramp uses smoothstep easing to avoid abrupt jumps.
    """
    T = len(x1)
    hold = int(max(1, hold))
    ramp = int(max(0, ramp))

    # center the hold window at mid
    hold_span = hold // 2
    h0, h1 = max(0, mid - hold_span), min(T, mid + hold_span + 1)

    # ramp in/out windows
    rin0, rin1 = max(0, h0 - ramp), h0
    rout0, rout1 = h1, min(T, h1 + ramp)

    # If ramp==0, this becomes a hard (teleport) overlap event.

    # ramp-in: blend from original -> overlap
    if ramp > 0 and rin1 > rin0:
        u = np.linspace(0, 1, rin1 - rin0, dtype=np.float32)
        a = _ease_in_out(u)
        x1[rin0:rin1] = x1[rin0:rin1] * (1 - a) + x_ov * a
        y1[rin0:rin1] = y1[rin0:rin1] * (1 - a) + y_ov * a
        x2[rin0:rin1] = x2[rin0:rin1] * (1 - a) + x_ov * a
        y2[rin0:rin1] = y2[rin0:rin1] * (1 - a) + y_ov * a

    # hold: exact overlap
    x1[h0:h1] = x_ov
    y1[h0:h1] = y_ov
    x2[h0:h1] = x_ov
    y2[h0:h1] = y_ov

    # ramp-out: blend overlap -> original
    if ramp > 0 and rout1 > rout0:
        u = np.linspace(0, 1, rout1 - rout0, dtype=np.float32)
        a = _ease_in_out(u)
        x1[rout0:rout1] = x_ov * (1 - a) + x1[rout0:rout1] * a
        y1[rout0:rout1] = y_ov * (1 - a) + y1[rout0:rout1] * a
        x2[rout0:rout1] = x_ov * (1 - a) + x2[rout0:rout1] * a
        y2[rout0:rout1] = y_ov * (1 - a) + y2[rout0:rout1] * a


def _fit_range(a: np.ndarray, low: float, high: float) -> np.ndarray:
    """Affine map array a into [low, high] preserving shape.

    If a has near-zero range, just clip.
    """
    amin = float(np.min(a))
    amax = float(np.max(a))
    if amax - amin < 1e-6:
        return np.clip(a, low, high)
    # scale to [0,1]
    u = (a - amin) / (amax - amin)
    return low + u * (high - low)


def _postprocess_tracks(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prevent hitting clamps by fitting into safe range before final clamp."""
    # Safe ranges (leave margins so later event edits don't clip)
    x_low, x_high = 0.08, 0.92
    y_low, y_high = 0.48, 0.86

    # Fit each object's y into range but keep relative shape
    y1 = _fit_range(y1, y_low, y_high)
    y2 = _fit_range(y2, y_low, y_high)

    # Fit x too (usually safe, but keep margin)
    x1 = _fit_range(x1, x_low, x_high)
    x2 = _fit_range(x2, x_low, x_high)

    return x1, y1, x2, y2


def _apply_min_separation(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    mid: int,
    hold: int,
    min_sep: float,
    rng: np.random.Generator,
):
    """Ensure that during a hold window, the two trajectories are not exactly identical.

    Reviewer-facing: avoid ill-posed "perfect merge for many frames".
    We keep a tiny offset (>=min_sep) to simulate partial occlusion rather than singularity.

    Only used in non-hard modes / non-cross single-frame events.
    """
    if hold <= 1 or min_sep <= 0:
        return

    T = len(x1)
    span = hold // 2
    s0, s1 = max(0, mid - span), min(T, mid + span + 1)

    # random offset direction, constant over the window
    ang = float(rng.uniform(0, 2 * np.pi))
    dx = float(np.cos(ang) * min_sep)
    dy = float(np.sin(ang) * min_sep)

    x2[s0:s1] = x2[s0:s1] + dx
    y2[s0:s1] = y2[s0:s1] + dy


def _clamp_tracks(tr1: np.ndarray, tr2: np.ndarray, x_min=0.05, x_max=0.95, y_min=0.45, y_max=0.88):
    """Final clamp to safe region."""
    tr1[:, 0] = np.clip(tr1[:, 0], x_min, x_max)
    tr2[:, 0] = np.clip(tr2[:, 0], x_min, x_max)
    tr1[:, 1] = np.clip(tr1[:, 1], y_min, y_max)
    tr2[:, 1] = np.clip(tr2[:, 1], y_min, y_max)
    return tr1, tr2


def make_tracks(
    task: str,
    variant_id: int,
    obj_type: str,
    bg_class: str,
    case_id: int,
    stress_mode: str = "none",
    overlap_frames: int = 9,
    smooth_window: int = 4,
    min_separation_px: float = 3.0,
    track_noise_std: float = 0.006,
) -> Tracks:
    """Generate trajectories.

    stress_mode:
      - "none": benchmark default (still jitter + non-symmetry)
      - "hard": destructive mode to force *exact overlap* for several frames

    overlap_frames:
      number of consecutive frames to force exact overlap (used in stress_mode=="hard")
    """
    # IMPORTANT: make trajectories case-specific to avoid repeated "template" motions.
    rng = seeded_rng("tracks", task, variant_id, obj_type, bg_class, case_id, stress_mode, overlap_frames)
    t = np.linspace(0, 1, NUM_FRAMES).astype(np.float32)
    overlap_frames = int(max(1, min(NUM_FRAMES, overlap_frames)))
    if overlap_frames % 2 == 0:
        overlap_frames += 1

    smooth_window = int(max(0, smooth_window))
    min_separation_px = float(max(0.0, min_separation_px))
    # convert pixel separation to normalized units (roughly isotropic)
    min_sep = float(min_separation_px / max(W, H))
    track_noise_std = float(max(0.0, track_noise_std))

    # keep trajectories in floor region (avoid hitting y clamp ceiling)
    y_center = rng.uniform(0.58, 0.68)

    if task == "cross":
        k = rng.uniform(0.16, 0.38) * (1.0 if rng.random() < 0.5 else -1.0)
        b = y_center

        # default: non-symmetric crossing point + phase shift
        db = rng.uniform(0.012, 0.030) * (1.0 if rng.random() < 0.5 else -1.0)
        phi1, phi2 = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)

        if stress_mode == "hard":
            # destructive: remove non-symmetry & remove phase shift
            db = 0.0
            phi1, phi2 = 0.0, 0.0

        b1, b2 = b + db, b - db

        t1 = t + 0.020 * np.sin(2 * np.pi * t + phi1)
        t2 = t + 0.020 * np.sin(2 * np.pi * t + phi2)
        t1 = np.clip(t1, 0, 1)
        t2 = np.clip(t2, 0, 1)

        # symmetric x paths (still allow tiny bias in non-hard mode)
        x1 = 0.12 + 0.78 * t1 + (0.0 if stress_mode == "hard" else rng.uniform(-0.03, 0.03))
        x2 = 0.88 - 0.78 * t2 + (0.0 if stress_mode == "hard" else rng.uniform(-0.03, 0.03))

        y1 = k * x1 + b1
        y2 = (-k) * x2 + b2

        # perturb only when not hard
        if stress_mode != "hard":
            y1 += _smooth_noise(rng, t, amp=rng.uniform(0.007, 0.016))
            y2 += _smooth_noise(rng, t, amp=rng.uniform(0.007, 0.016))
            x1 += _smooth_noise(rng, t, amp=rng.uniform(0.004, 0.012))
            x2 += _smooth_noise(rng, t, amp=rng.uniform(0.004, 0.012))

        # fit into safe range first (prevents ceiling clamping), then apply overlap event
        x1, y1, x2, y2 = _postprocess_tracks(x1, y1, x2, y2)

        # guarantee CROSS happens (but without teleportation)
        mid = NUM_FRAMES // 2
        x_ov = float(np.clip(0.50 + rng.uniform(-0.02, 0.02), 0.10, 0.90))
        y_ov = float(np.clip(b + rng.uniform(-0.01, 0.01), 0.55, 0.85))

        if stress_mode == "hard":
            hold = overlap_frames
            ramp = max(smooth_window, overlap_frames // 2) if smooth_window > 0 else 0
        else:
            hold = 1
            ramp = smooth_window

        _apply_overlap_event(x1, y1, x2, y2, mid, x_ov, y_ov, hold=hold, ramp=ramp)
        # In default mode hold=1; in hard mode allow perfect overlap.
        # For cross in default mode, hold=1 so no singularity; in hard mode we allow perfect overlap on purpose.

    elif task == "parallel":
        # guarantee PARALLEL happens (never cross / never occlude):
        # - keep same x motion
        # - keep a minimum y gap, and apply the SAME jitter to both
        # Note: no overlap event here.
        x_base = 0.15 + 0.70 * t + rng.uniform(-0.02, 0.02)
        x1 = x_base + _smooth_noise(rng, t, amp=rng.uniform(0.003, 0.008))
        x2 = x_base + _smooth_noise(rng, t, amp=rng.uniform(0.003, 0.008))

        y_gap = float(rng.uniform(0.10, 0.22))
        y_gap = max(y_gap, 0.08)

        y1 = y_center - y_gap / 2
        y2 = y_center + y_gap / 2

        # same low-freq jitter (preserves parallelism)
        jy = _smooth_noise(rng, t, amp=rng.uniform(0.004, 0.010))
        y1 = y1 + jy
        y2 = y2 + jy

        # ensure minimum separation at all frames
        min_gap = 0.06
        gap_now = (y2 - y1)
        if np.min(gap_now) < min_gap:
            y2 = y2 + (min_gap - np.min(gap_now))

        # fit into safe range (keeps parallelism)
        x1, y1, x2, y2 = _postprocess_tracks(x1, y1, x2, y2)

    elif task == "occlusion":
        # one static, one passes through
        x0 = rng.uniform(0.55, 0.75)
        y0 = y_center

        tr_static = np.stack([np.full_like(t, x0), np.full_like(t, y0)], 1)

        x_start = rng.uniform(0.10, 0.25)
        x_end = rng.uniform(0.80, 0.95)
        x_m = x_start + (x_end - x_start) * t

        y_m = np.full_like(t, y0)
        y_m += _smooth_noise(rng, t, amp=rng.uniform(0.006, 0.015))
        x_m += _smooth_noise(rng, t, amp=rng.uniform(0.004, 0.012))

        # enforce pass-through at mid
        mid = NUM_FRAMES // 2
        if stress_mode == "hard":
            miss = 0.0
        else:
            # still allow randomness, but occlusion must happen anyway
            miss = (variant_id % 3 - 1) * rng.uniform(0.000, 0.010)  # {-eps, 0, +eps}

        y_m += (y0 - y_m[mid]) + miss
        x_m += (x0 - x_m[mid])

        # set arrays for postprocess
        x1, y1 = tr_static[:, 0].copy(), tr_static[:, 1].copy()
        x2, y2 = x_m.copy(), y_m.copy()

        # fit into safe range first, then guarantee OCCLUSION happens (but without teleportation)
        x1, y1, x2, y2 = _postprocess_tracks(x1, y1, x2, y2)

        # update static anchor and mover after postprocess
        x0, y0 = float(x1[mid]), float(y1[mid])
        x_m, y_m = x2.copy(), y2.copy()

        # - default: short hold (3 frames) with ramp
        # - hard: longer hold (overlap_frames) with ramp
        hold = overlap_frames if stress_mode == "hard" else 3
        ramp = max(smooth_window, hold // 2) if smooth_window > 0 else 0
        _apply_overlap_event(x_m, y_m, x_m, y_m, mid, x0, y0, hold=hold, ramp=ramp)
        # Avoid ill-posed multi-frame perfect merge in default mode
        if stress_mode != "hard":
            _apply_min_separation(x1, y1, x_m, y_m, mid=mid, hold=hold, min_sep=min_sep, rng=rng)

        # write back
        x2, y2 = x_m, y_m

        # ensure static is still constant (Object A)
        x1[:] = x0
        y1[:] = y0

    elif task == "attribute_confusion":
        # cross-like but gentler; keeps attention conflict while focusing on attributes
        k = rng.uniform(0.10, 0.22)
        b = rng.uniform(0.62, 0.70)

        phi1, phi2 = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
        if stress_mode == "hard":
            phi1, phi2 = 0.0, 0.0
        t1 = t + 0.014 * np.sin(2 * np.pi * t + phi1)
        t2 = t + 0.014 * np.sin(2 * np.pi * t + phi2)
        t1 = np.clip(t1, 0, 1)
        t2 = np.clip(t2, 0, 1)

        x1 = 0.18 + 0.64 * t1
        x2 = 0.82 - 0.64 * t2

        db = rng.uniform(0.008, 0.018) * (1.0 if rng.random() < 0.5 else -1.0)
        if stress_mode == "hard":
            db = 0.0
        y1 = k * x1 + (b + db)
        y2 = (-k) * x2 + (b - db)

        y1 += _smooth_noise(rng, t, amp=rng.uniform(0.004, 0.010))
        y2 += _smooth_noise(rng, t, amp=rng.uniform(0.004, 0.010))

        # fit into safe range first, then guarantee an attention-conflict event (cross/overlap)
        x1, y1, x2, y2 = _postprocess_tracks(x1, y1, x2, y2)

        mid = NUM_FRAMES // 2
        x_ov = float(np.clip(0.50 + rng.uniform(-0.02, 0.02), 0.10, 0.90))
        y_ov = float(np.clip(b + rng.uniform(-0.01, 0.01), 0.55, 0.85))

        if stress_mode == "hard":
            hold = overlap_frames
            ramp = max(smooth_window, overlap_frames // 2) if smooth_window > 0 else 0
        else:
            hold = 1
            ramp = smooth_window

        _apply_overlap_event(x1, y1, x2, y2, mid, x_ov, y_ov, hold=hold, ramp=ramp)

    elif task == "identity_stress":
        # A dedicated failure-mining trajectory:
        # 1) start as parallel
        # 2) converge to near overlap
        # 3) braid / interweave around a center line
        # 4) optionally force exact overlap window in hard mode
        x_mid = rng.uniform(0.45, 0.55)
        y_mid = rng.uniform(0.60, 0.72)

        # forward motion
        x_base = 0.15 + 0.70 * t

        # braid offsets
        # large enough to confuse attention, but still plausible
        amp_y = rng.uniform(0.06, 0.10)
        amp_x = rng.uniform(0.03, 0.06)
        phase = rng.uniform(0, 2 * np.pi)

        # two trajectories weave in opposite phase
        weave = np.sin(2 * np.pi * t * 1.2 + phase)
        x1 = x_base + amp_x * weave
        x2 = x_base - amp_x * weave

        y1 = y_mid - amp_y * weave
        y2 = y_mid + amp_y * weave

        # add mild jitter
        y1 += _smooth_noise(rng, t, amp=rng.uniform(0.004, 0.010))
        y2 += _smooth_noise(rng, t, amp=rng.uniform(0.004, 0.010))
        x1 += _smooth_noise(rng, t, amp=rng.uniform(0.003, 0.008))
        x2 += _smooth_noise(rng, t, amp=rng.uniform(0.003, 0.008))

        if stress_mode == "hard":
            mid = NUM_FRAMES // 2
            span = overlap_frames // 2
            s0, s1 = max(0, mid - span), min(NUM_FRAMES, mid + span + 1)
            x1[s0:s1] = x_mid
            x2[s0:s1] = x_mid
            y1[s0:s1] = y_mid
            y2[s0:s1] = y_mid

    # Add mild asymmetry / realism noise (low amplitude) AFTER events.
    # Use low-frequency noise so it doesn't look like pixel jitter.
    if track_noise_std > 0 and stress_mode != "hard":
        jx1 = _smooth_noise(rng, t, amp=track_noise_std * float(rng.uniform(0.6, 1.1)))
        jy1 = _smooth_noise(rng, t, amp=track_noise_std * float(rng.uniform(0.6, 1.1)))
        jx2 = _smooth_noise(rng, t, amp=track_noise_std * float(rng.uniform(0.6, 1.1)))
        jy2 = _smooth_noise(rng, t, amp=track_noise_std * float(rng.uniform(0.6, 1.1)))
        x1 = x1 + jx1
        y1 = y1 + jy1
        x2 = x2 + jx2
        y2 = y2 + jy2

    tr1 = np.stack([x1, y1], 1)
    tr2 = np.stack([x2, y2], 1)

    # final clamp
    tr1, tr2 = _clamp_tracks(tr1, tr2, y_max=0.88)

    return Tracks(tr1=tr1.astype(np.float32), tr2=tr2.astype(np.float32), meta={"variant_id": variant_id})


# ----------------------------
# Rendering primitives
# ----------------------------

def draw_shadow(img: np.ndarray, x: int, y: int, r: int):
    cv2.ellipse(img, (x, y + r), (r, int(r * 0.30)), 0, 0, 360, (0, 0, 0), -1)


def draw_sphere(img: np.ndarray, center: Tuple[int, int], r: int, color_bgr, light_dir: str):
    x0, y0 = center
    light = get_light_vector(light_dir)
    base = np.array(color_bgr, dtype=np.float32)

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            x = x0 + dx
            y = y0 + dy
            if x < 0 or x >= W or y < 0 or y >= H:
                continue

            nx, ny = dx / r, dy / r
            nz = np.sqrt(max(0.0, 1 - nx * nx - ny * ny))
            normal = np.array([nx, ny, nz], dtype=np.float32)

            ambient = 0.22
            diffuse = max(float(np.dot(normal, light)), 0.0)

            view = np.array([0, 0, 1], dtype=np.float32)
            reflect = 2 * np.dot(normal, light) * normal - light
            spec = max(float(np.dot(reflect, view)), 0.0) ** 12

            intensity = ambient + 0.80 * diffuse + 0.20 * spec
            col = np.clip(base * intensity, 0, 255)
            img[y, x] = col


def draw_ellipsoid(img: np.ndarray, center: Tuple[int, int], rx: int, ry: int, color_bgr, light_dir: str):
    x0, y0 = center
    light = get_light_vector(light_dir)
    base = np.array(color_bgr, dtype=np.float32)

    for dy in range(-ry, ry + 1):
        for dx in range(-rx, rx + 1):
            if (dx * dx) / (rx * rx + 1e-6) + (dy * dy) / (ry * ry + 1e-6) > 1.0:
                continue
            x = x0 + dx
            y = y0 + dy
            if x < 0 or x >= W or y < 0 or y >= H:
                continue

            nx = dx / (rx + 1e-6)
            ny = dy / (ry + 1e-6)
            nz = np.sqrt(max(0.0, 1 - nx * nx - ny * ny))
            normal = np.array([nx, ny, nz], dtype=np.float32)

            ambient = 0.22
            diffuse = max(float(np.dot(normal, light)), 0.0)
            view = np.array([0, 0, 1], dtype=np.float32)
            reflect = 2 * np.dot(normal, light) * normal - light
            spec = max(float(np.dot(reflect, view)), 0.0) ** 12

            intensity = ambient + 0.80 * diffuse + 0.20 * spec
            col = np.clip(base * intensity, 0, 255)
            img[y, x] = col


def draw_prism(img: np.ndarray, center: Tuple[int, int], r: int, color_bgr, aspect=(1.0, 1.0), angle_deg=0.0):
    x, y = center
    base = np.array(color_bgr, dtype=np.float32)
    front = base * 0.90
    top = base * 1.15
    side = base * 0.60

    rx = int(r * aspect[0])
    ry = int(r * aspect[1])
    offset = int(r * 0.55)

    top_pts = np.array([[-rx, -ry], [rx, -ry], [rx + offset, -ry - offset], [-rx + offset, -ry - offset]], dtype=np.float32)
    side_pts = np.array([[rx, -ry], [rx + offset, -ry - offset], [rx + offset, ry - offset], [rx, ry]], dtype=np.float32)
    front_rect = np.array([[-rx, -ry], [rx, -ry], [rx, ry], [-rx, ry]], dtype=np.float32)

    if abs(angle_deg) > 1e-3:
        top_pts = rot2d(top_pts, angle_deg)
        side_pts = rot2d(side_pts, angle_deg)
        front_rect = rot2d(front_rect, angle_deg)

    top_pts = (top_pts + np.array([x, y], dtype=np.float32)).astype(int)
    side_pts = (side_pts + np.array([x, y], dtype=np.float32)).astype(int)
    front_rect = (front_rect + np.array([x, y], dtype=np.float32)).astype(int)

    cv2.fillPoly(img, [top_pts], np.clip(top, 0, 255).astype(np.uint8).tolist())
    cv2.fillPoly(img, [front_rect], np.clip(front, 0, 255).astype(np.uint8).tolist())
    cv2.fillPoly(img, [side_pts], np.clip(side, 0, 255).astype(np.uint8).tolist())


def draw_arrow(img: np.ndarray, center: Tuple[int, int], r: int, color_bgr, angle_deg=0.0, thickness=1.0):
    x, y = center
    col = np.array(color_bgr, dtype=np.float32)
    col1 = np.clip(col * 1.10, 0, 255).astype(np.uint8).tolist()
    col2 = np.clip(col * 0.75, 0, 255).astype(np.uint8).tolist()

    body_w = int(r * 0.62)
    body_h = max(2, int(r * 0.26 * thickness))
    head_h = max(3, int(r * 0.55 * thickness))

    body = np.array([[-body_w, -body_h], [body_w, -body_h], [body_w, body_h], [-body_w, body_h]], dtype=np.float32)
    head = np.array([[body_w, -head_h], [int(r * 1.20), 0], [body_w, head_h]], dtype=np.float32)

    if abs(angle_deg) > 1e-3:
        body = rot2d(body, angle_deg)
        head = rot2d(head, angle_deg)

    body = (body + np.array([x, y], dtype=np.float32)).astype(int)
    head = (head + np.array([x, y], dtype=np.float32)).astype(int)

    cv2.fillPoly(img, [body], col2)
    cv2.fillPoly(img, [head], col1)
    cv2.polylines(img, [body], True, col1, 2, cv2.LINE_AA)


# ----------------------------
# Object specs
# ----------------------------

def object_specs(task: str, obj_type: str, case_id: int, variant_id: int):
    """Return specs for Object A (track1) and Object B (track2). Deterministic.

    Note: specs may include optional keys:
      - material: "metal" | "plastic" (used mainly for prompt semantics)
    """

    if task == "attribute_confusion":
        # Strengthened killer: mix several "semantic-close + visual-conflict" presets.
        # We keep it deterministic by variant_id.
        rng = seeded_rng("attr", obj_type, variant_id)

        if obj_type == "sphere":
            mode = int(variant_id % 3)
            if mode == 0:
                # original killer: light/deep blue + circle/ellipse
                return (
                    {"kind": "circle", "color": "light_blue", "base_r": 46, "material": "metal"},
                    {"kind": "ellipse", "color": "deep_blue", "base_r": 46, "material": "metal"},
                )
            if mode == 1:
                # semantic-close colors (red vs orange), same shape
                return (
                    {"kind": "sphere", "color": "red", "base_r": 46, "material": "metal"},
                    {"kind": "sphere", "color": "orange", "base_r": 46, "material": "metal"},
                )
            # mode == 2
            # same color, subtle geometry (circle vs slight-ellipse) to force binding over classification
            return (
                {"kind": "circle", "color": "yellow", "base_r": 46, "material": "plastic"},
                {"kind": "ellipse", "color": "yellow", "base_r": 46, "material": "plastic"},
            )

        # structured: rotate between (prism vs cube), (thin vs thick arrow), (material ambiguity)
        mode = int(variant_id % 3)
        angle = float(rng.uniform(-25, 25))

        if mode == 0:
            return (
                {"kind": "prism", "color": "light_blue", "base_r": 46, "aspect": (1.18, 0.88), "angle": angle, "material": "metal"},
                {"kind": "cube", "color": "deep_blue", "base_r": 46, "aspect": (1.00, 1.00), "angle": angle, "material": "metal"},
            )
        if mode == 1:
            angle = float(rng.uniform(-30, 30))
            return (
                {"kind": "arrow", "color": "light_blue", "base_r": 46, "angle": angle, "thickness": 0.78, "material": "plastic"},
                {"kind": "arrow", "color": "deep_blue", "base_r": 46, "angle": angle, "thickness": 1.00, "material": "plastic"},
            )

        # mode == 2: CLIP-level ambiguity: same color, same shape, only material differs
        same_c = str(rng.choice(["blue", "purple", "orange"]))
        return (
            {"kind": "cube", "color": same_c, "base_r": 46, "aspect": (1.00, 1.00), "angle": angle, "material": "metal"},
            {"kind": "cube", "color": same_c, "base_r": 46, "aspect": (1.00, 1.00), "angle": -angle, "material": "plastic"},
        )

    rng = seeded_rng("objs", task, obj_type, case_id)

    if obj_type == "sphere":
        palette = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]
        c1 = str(rng.choice(palette))
        c2 = str(rng.choice([x for x in palette if x != c1]))
        return (
            {"kind": "sphere", "color": c1, "base_r": int(rng.integers(42, 52))},
            {"kind": "sphere", "color": c2, "base_r": int(rng.integers(42, 52))},
        )

    # structured
    palette = ["red", "blue", "green", "yellow", "purple", "orange"]
    c1 = str(rng.choice(palette))
    c2 = str(rng.choice([x for x in palette if x != c1]))

    angle = float(rng.uniform(-25, 25))
    if (case_id % 3) == 0:
        a = (float(rng.uniform(0.95, 1.25)), float(rng.uniform(0.80, 1.10)))
        return (
            {"kind": "prism", "color": c1, "base_r": int(rng.integers(44, 54)), "aspect": a, "angle": angle},
            {"kind": "cube", "color": c2, "base_r": int(rng.integers(44, 54)), "aspect": (1.0, 1.0), "angle": -angle},
        )
    if (case_id % 3) == 1:
        return (
            {"kind": "arrow", "color": c1, "base_r": int(rng.integers(44, 54)), "angle": angle, "thickness": 1.0},
            {"kind": "cube", "color": c2, "base_r": int(rng.integers(44, 54)), "aspect": (1.0, 1.0), "angle": -angle},
        )
    return (
        {"kind": "cube", "color": c1, "base_r": int(rng.integers(44, 54)), "aspect": (1.0, 1.0), "angle": angle},
        {"kind": "arrow", "color": c2, "base_r": int(rng.integers(44, 54)), "angle": -angle, "thickness": 1.0},
    )


# ----------------------------
# First-frame rendering
# ----------------------------

def render_first_frame(bg_img: np.ndarray, light_dir: str, tracks: Tracks, objA: Dict, objB: Dict) -> np.ndarray:
    img = bg_img.copy()

    x1, y1 = int(tracks.tr1[0, 0] * W), int(tracks.tr1[0, 1] * H)
    x2, y2 = int(tracks.tr2[0, 0] * W), int(tracks.tr2[0, 1] * H)

    # depth: larger y is closer -> draw back first
    order = [(0, x1, y1, objA), (1, x2, y2, objB)]
    order.sort(key=lambda z: z[2])

    for _, x, y, spec in order:
        r = int(spec["base_r"] * get_scale(y))
        draw_shadow(img, x, y, r)

        col = BGR[spec["color"]]
        kind = spec["kind"]

        if kind in ("sphere", "circle"):
            draw_sphere(img, (x, y), r, col, light_dir)
        elif kind == "ellipse":
            draw_ellipsoid(img, (x, y), r, int(r * 0.72), col, light_dir)
        elif kind == "cube":
            draw_prism(img, (x, y), r, col, aspect=(1.0, 1.0), angle_deg=float(spec.get("angle", 0.0)))
        elif kind == "prism":
            draw_prism(img, (x, y), r, col, aspect=tuple(spec.get("aspect", (1.0, 1.0))), angle_deg=float(spec.get("angle", 0.0)))
        else:  # arrow
            draw_arrow(img, (x, y), r, col, angle_deg=float(spec.get("angle", 0.0)), thickness=float(spec.get("thickness", 1.0)))

    return img


# ----------------------------
# Visibility (smoothed)
# ----------------------------

def effective_radius(spec: Dict, y_px: int) -> float:
    base = float(spec["base_r"]) * float(get_scale(y_px))
    kind = spec["kind"]

    if kind in ("sphere", "circle"):
        return base
    if kind == "ellipse":
        return base
    if kind in ("cube", "prism"):
        ar = spec.get("aspect", (1.0, 1.0))
        return base * max(ar[0], ar[1]) * 1.05
    if kind == "arrow":
        return base * 1.10
    return base


def compute_visibility(tracks: Tracks, objA: Dict, objB: Dict) -> np.ndarray:
    """Smoothed visibility.

    Raw overlap + y-depth, then stabilize with a 3-frame window.
    """
    raw = np.ones((2, NUM_FRAMES), dtype=np.float32)

    for k in range(NUM_FRAMES):
        x1, y1 = int(tracks.tr1[k, 0] * W), int(tracks.tr1[k, 1] * H)
        x2, y2 = int(tracks.tr2[k, 0] * W), int(tracks.tr2[k, 1] * H)

        rr1 = effective_radius(objA, y1)
        rr2 = effective_radius(objB, y2)

        d = float(np.hypot(x1 - x2, y1 - y2))
        thr = 0.92 - 0.08 * np.tanh((rr1 + rr2) / 120.0)

        if d < (rr1 + rr2) * thr:
            if y1 < y2:
                raw[0, k] = 0.0
            elif y2 < y1:
                raw[1, k] = 0.0
            else:
                raw[0, k] = 0.0

    vis = raw.copy()
    for obj_idx in [0, 1]:
        occ = (raw[obj_idx] < 0.5).astype(np.int32)
        occ_pad = np.pad(occ, (2, 0), mode="edge")
        roll = occ_pad[2:] + occ_pad[1:-1] + occ_pad[:-2]
        occ_smooth = (roll >= 2).astype(np.float32)
        vis[obj_idx] = 1.0 - occ_smooth

    return vis.astype(np.float32)


# ----------------------------
# Prompt
# ----------------------------

def color_word(color_key: str) -> str:
    if color_key == "light_blue":
        return "Light blue"
    if color_key == "deep_blue":
        return "Deep blue"
    return color_key.capitalize()


def shape_phrase(spec: Dict, rng: np.random.Generator) -> str:
    k = spec["kind"]
    if k in ("sphere", "circle"):
        return str(rng.choice(["orb", "sphere", "ball"]))
    if k == "ellipse":
        return str(rng.choice(["elliptical orb", "ellipse", "oval sphere"]))
    if k == "cube":
        return str(rng.choice(["cube", "block"]))
    if k == "prism":
        return str(rng.choice(["rectangular prism", "stretched cube"]))
    return str(rng.choice(["arrow", "dart"]))


def material_phrase(spec: Dict) -> str:
    m = spec.get("material", None)
    if m is None:
        return ""
    if m == "metal":
        return "metal"
    if m == "plastic":
        return "plastic"
    return str(m)


def build_prompt(task: str, bg_variant: str, objA: Dict, objB: Dict, case_id: int) -> str:
    rng = seeded_rng("prompt", case_id)

    mA = material_phrase(objA)
    mB = material_phrase(objB)

    # Put material near the noun (CLIP-level ambiguity), but avoid awkward duplicates like "metallic metal".
    styleA = str(rng.choice(STYLE_WORDS))
    styleB = str(rng.choice(STYLE_WORDS))
    if styleA == "metallic" and mA == "metal":
        mA = ""
    if styleB == "metallic" and mB == "metal":
        mB = ""

    sA = f"{color_word(objA['color'])} {styleA} {mA + ' ' if mA else ''}{shape_phrase(objA, rng)}".strip()
    sB = f"{color_word(objB['color'])} {styleB} {mB + ' ' if mB else ''}{shape_phrase(objB, rng)}".strip()

    return (
        f"Object A: {sA}.\n"
        f"Object B: {sB}.\n"
        f"They move with a {task} trajectory. Background: {bg_variant}."
    )


# ----------------------------
# Main generation
# ----------------------------

def generate_dataset(
    save_root: str = DEFAULT_SAVE_ROOT,
    n_variants: int = 15,
    attribute_variants: Optional[int] = None,
    keep_meta: bool = True,
    stress_mode: str = "none",
    overlap_frames: int = 9,
    include_identity_stress: bool = False,
    smooth_window: int = 4,
    same_color_ratio: float = 0.30,
    record_motion_stats: bool = True,
    min_separation_px: float = 3.0,
    track_noise_std: float = 0.006,
):
    if attribute_variants is None:
        attribute_variants = n_variants

    smooth_window = int(max(0, smooth_window))
    same_color_ratio = float(np.clip(same_color_ratio, 0.0, 1.0))

    # compute total
    task_list = list(TASKS) + (list(OPTIONAL_TASKS) if include_identity_stress else [])

    total = 0
    for task in task_list:
        v = attribute_variants if task == "attribute_confusion" else n_variants
        total += len(OBJECT_TYPES) * len(BG_CLASSES) * v

    # Pre-compute EXACT same-color allocation per (task,obj_type,bg_class) group.
    # This gives deterministic, balanced proportions (e.g., 30% same-color, 70% different-color).
    same_color_sets = {}
    for task in task_list:
        if task == "attribute_confusion":
            # attribute_confusion is fixed to a designed color pair; don't override.
            continue
        vcount = attribute_variants if task == "attribute_confusion" else n_variants
        for obj_type in OBJECT_TYPES:
            for bg_class in BG_CLASSES:
                rng_sc = seeded_rng("same_color", task, obj_type, bg_class, vcount, same_color_ratio)
                ids = list(range(vcount))
                rng_sc.shuffle(ids)
                n_same = int(round(same_color_ratio * vcount))
                same_color_sets[(task, obj_type, bg_class)] = set(ids[:n_same])

    # motion stats accumulator
    motion_stats = []  # list of dicts

    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.makedirs(save_root, exist_ok=True)

    case_id = 0

    for task in task_list:
        vcount = attribute_variants if task == "attribute_confusion" else n_variants

        for obj_type in OBJECT_TYPES:
            for bg_class in BG_CLASSES:
                for variant_id in range(vcount):
                    case_path = os.path.join(save_root, f"case_{case_id:04d}")
                    os.makedirs(case_path, exist_ok=True)

                    light_dir = pick_light(task, obj_type, bg_class, variant_id)
                    bg_variant, bg_img, bg_seed = choose_background(bg_class, task, obj_type, variant_id, case_id, light_dir)

                    tracks = make_tracks(
                        task,
                        variant_id,
                        obj_type=obj_type,
                        bg_class=bg_class,
                        case_id=case_id,
                        stress_mode=stress_mode,
                        overlap_frames=overlap_frames,
                        smooth_window=smooth_window,
                        min_separation_px=min_separation_px,
                        track_noise_std=track_noise_std,
                    )
                    objA, objB = object_specs(task, obj_type, case_id, variant_id)

                    # Color diversity control (except attribute_confusion):
                    # enforce a fixed proportion of same-color cases.
                    if task != "attribute_confusion":
                        same_color = variant_id in same_color_sets.get((task, obj_type, bg_class), set())
                        rng_c = seeded_rng("colors", task, obj_type, bg_class, case_id)
                        if obj_type == "sphere":
                            palette = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]
                        else:
                            palette = ["red", "blue", "green", "yellow", "purple", "orange"]
                        c = str(rng_c.choice(palette))
                        if same_color:
                            objA["color"] = c
                            objB["color"] = c
                        else:
                            c2 = str(rng_c.choice([x for x in palette if x != c]))
                            objA["color"] = c
                            objB["color"] = c2

                    img = render_first_frame(bg_img, light_dir, tracks, objA, objB)
                    vis = compute_visibility(tracks, objA, objB)
                    prompt = build_prompt(task, bg_variant, objA, objB, case_id)

                    # Motion difficulty label: max frame-to-frame displacement (per object and max of two)
                    if record_motion_stats:
                        dA = np.linalg.norm(np.diff(tracks.tr1, axis=0), axis=1)
                        dB = np.linalg.norm(np.diff(tracks.tr2, axis=0), axis=1)
                        max_vel_A = float(dA.max()) if len(dA) else 0.0
                        max_vel_B = float(dB.max()) if len(dB) else 0.0
                        max_vel = max(max_vel_A, max_vel_B)
                        motion_stats.append({
                            "case_id": case_id,
                            "task": task,
                            "object_type": obj_type,
                            "bg_class": bg_class,
                            "variant_id": variant_id,
                            "max_velocity": max_vel,
                            "max_velocity_A": max_vel_A,
                            "max_velocity_B": max_vel_B,
                        })
                    else:
                        max_vel = None
                        max_vel_A = None
                        max_vel_B = None

                    cv2.imwrite(os.path.join(case_path, "image.jpg"), img)
                    np.save(os.path.join(case_path, "tracks.npy"), np.stack([tracks.tr1, tracks.tr2]))
                    np.save(os.path.join(case_path, "visibility.npy"), vis)

                    with open(os.path.join(case_path, "prompt.txt"), "w", encoding="utf-8") as f:
                        f.write(prompt)

                    if keep_meta:
                        meta = {
                            "case_id": case_id,
                            "task": task,
                            "object_type": obj_type,
                            "bg_class": bg_class,
                            "bg_variant": bg_variant,
                            "bg_seed": bg_seed,
                            "variant_id": variant_id,
                            "light_dir": light_dir,
                            "stress_mode": stress_mode,
                            "overlap_frames": overlap_frames,
                            "smooth_window": smooth_window,
                            "same_color_ratio": same_color_ratio,
                            "entities": {"A": objA, "B": objB},
                            "motion": {
                                "max_velocity": max_vel,
                                "max_velocity_A": max_vel_A,
                                "max_velocity_B": max_vel_B,
                            },
                            "files": {
                                "image": "image.jpg",
                                "tracks": "tracks.npy",
                                "visibility": "visibility.npy",
                                "prompt": "prompt.txt",
                            },
                        }
                        with open(os.path.join(case_path, "meta.json"), "w", encoding="utf-8") as f:
                            json.dump(meta, f, ensure_ascii=False, indent=2)

                    case_id += 1

    # Save global motion stats summary (optional)
    if record_motion_stats and len(motion_stats) > 0:
        ms = np.array([m["max_velocity"] for m in motion_stats], dtype=np.float32)
        q = {
            "p10": float(np.quantile(ms, 0.10)),
            "p50": float(np.quantile(ms, 0.50)),
            "p90": float(np.quantile(ms, 0.90)),
        }
        # Suggested bins: data-driven (quantiles) to avoid over-claiming.
        # You can still set fixed thresholds in your paper later.
        summary = {
            "total_cases": int(len(motion_stats)),
            "max_velocity_quantiles": q,
            "suggested_bins": {
                "low": f"<= {q['p10']:.4f}",
                "mid": f"({q['p10']:.4f}, {q['p90']:.4f}]",
                "high": f"> {q['p90']:.4f}",
            },
            "note": "Bins are dataset-quantile based; report raw max_velocity for rigor.",
        }
        with open(os.path.join(save_root, "motion_stats_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(save_root, "motion_stats_per_case.json"), "w", encoding="utf-8") as f:
            json.dump(motion_stats, f, ensure_ascii=False)

    print(f"✅ EntityBench 生成完成 ({total} cases) -> {save_root}")


def _parse_args():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--save_root", type=str, default=DEFAULT_SAVE_ROOT)
    p.add_argument("--n_variants", type=int, default=15)
    p.add_argument(
        "--attribute_variants",
        type=int,
        default=None,
        help="override variants for attribute_confusion only (default: same as n_variants)",
    )
    p.add_argument(
        "--lite180",
        action="store_true",
        help="Generate ~180 cases: keep 15 variants for 3 tasks, but only 5 variants for attribute_confusion.",
    )
    p.add_argument("--no_meta", action="store_true")

    # Trajectory stress options (for failure mining)
    p.add_argument(
        "--stress_mode",
        type=str,
        default="none",
        choices=["none", "hard"],
        help="hard=long exact overlap; none=short overlap but physically smoothed",
    )
    p.add_argument(
        "--overlap_frames",
        type=int,
        default=9,
        help="consecutive frames forced to exact overlap when stress_mode=hard (odd number recommended)",
    )

    # Smoothing toggle: 0 means NO smoothing (teleport); >0 means smooth ramps.
    p.add_argument(
        "--smooth_window",
        type=int,
        default=4,
        help="ramp frames for overlap events. 0=teleport, 4=default smooth. Useful to separate discontinuity vs binding failures.",
    )
    p.add_argument(
        "--min_separation_px",
        type=float,
        default=3.0,
        help="In non-hard occlusion, keep a tiny separation (>= this many pixels) to avoid ill-posed multi-frame perfect merges.",
    )
    p.add_argument(
        "--track_noise_std",
        type=float,
        default=0.006,
        help="Extra low-frequency noise std (normalized units) to break perfect symmetry. Suggest 0.006~0.012.",
    )

    # Color diversity control
    p.add_argument(
        "--same_color_ratio",
        type=float,
        default=0.30,
        help="exact proportion of same-color cases per (task,obj_type,bg_class) group (except attribute_confusion).",
    )

    # Motion stats
    p.add_argument(
        "--no_motion_stats",
        action="store_true",
        help="disable writing motion_stats_summary.json and max_velocity fields",
    )

    # Optional extra task
    p.add_argument(
        "--include_identity_stress",
        action="store_true",
        help="Add optional identity_stress task (not part of the standard 4×2×2×15 benchmark).",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    attr_v = args.attribute_variants
    if args.lite180:
        # (3 tasks × 2 × 2 × 15) + (attribute × 2 × 2 × 5) = 180
        attr_v = 5

    generate_dataset(
        save_root=args.save_root,
        n_variants=args.n_variants,
        attribute_variants=attr_v,
        keep_meta=not args.no_meta,
        stress_mode=args.stress_mode,
        overlap_frames=args.overlap_frames,
        include_identity_stress=args.include_identity_stress,
        smooth_window=args.smooth_window,
        same_color_ratio=args.same_color_ratio,
        record_motion_stats=(not args.no_motion_stats),
        min_separation_px=args.min_separation_px,
        track_noise_std=args.track_noise_std,
    )
