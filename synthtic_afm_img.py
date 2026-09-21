#!/usr/bin/env python3
"""
Synthetic AFM heightmap generator for YOLO-seg (rectangles + triangles).

Pipeline per image
  1. sample shapes (class balanced by instance count) and place them
     (separate, touching or overlapping)
  2. optionally break shapes (bites / cut corners) -> completeness ground truth
  3. compose a clean heightmap in nm (each instance keeps its own FULL mask)
  4. make it look like AFM (tip blur, tilt + smooth background, noise,
     scan-line offsets/streaks, optional row-median flattening)
  5. save 8-bit PNG + YOLO-seg label + JSON metadata (ideal polygon, completeness)

Usage
  python generate_afm_data.py --out dataset --n-train 1000 --n-val 100 --preview 20

Adding a class later: append it to CLASS_NAMES and add a branch in
`ideal_polygon`. Never reorder existing class IDs.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import disk as draw_disk
from skimage.draw import polygon as draw_polygon
from skimage.measure import label as cc_label

# Class IDs are the list index. Append only, never renumber.
CLASS_NAMES = ["rectangle", "triangle"]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # --- image / physical conventions -------------------------------------- #
    img_size: int = 512
    px_nm: float = 5.0                    # metadata only (1 px = 5 nm)

    # --- dataset composition ----------------------------------------------- #
    negative_frac: float = 0.07           # background-only images (empty labels)
    n_shapes: tuple = (1, 15)             # instances per image (inclusive)
    touching_image_prob: float = 0.4      # images where touching/overlap allowed

    # --- shape geometry (pixels) ------------------------------------------- #
    rect_short_px: tuple = (15, 50)
    rect_aspect: tuple = (1.0, 3.0)
    tri_radius_px: tuple = (18, 48)       # circumradius
    tri_equilateral_prob: float = 0.5
    edge_margin: int = 3                  # keep shapes fully inside the image
    min_area_px: int = 80

    # --- placement ---------------------------------------------------------- #
    max_place_attempts: int = 60
    separate_gap_px: int = 3              # min gap in "separate" images
    max_overlap: float = 0.35             # max overlapped fraction in "touching" images
    anchor_prob: float = 0.7              # touching images: place next to existing shape
    touch_dist_factor: tuple = (0.55, 1.0)

    # --- incompleteness ----------------------------------------------------- #
    broken_prob: float = 0.35
    min_completeness: float = 0.5
    max_broken_completeness: float = 0.97

    # --- heights (nm) ------------------------------------------------------- #
    shape_height_nm: tuple = (1.0, 3.0)
    in_shape_var: float = 0.05            # relative smooth variation on top of shapes

    # --- AFM realism (ranges are sampled per image) ------------------------- #
    tip_sigma_px: tuple = (0.8, 2.0)
    tilt_nm: tuple = (0.0, 1.5)           # max height change across the image
    smooth_bg_amp_nm: tuple = (0.05, 0.5)
    smooth_bg_sigma_px: tuple = (30, 80)
    noise_sigma_nm: tuple = (0.03, 0.25)
    row_walk_step_nm: tuple = (0.005, 0.03)
    row_jitter_nm: tuple = (0.0, 0.08)
    streak_prob: float = 0.5
    streak_count: tuple = (1, 6)
    streak_amp_nm: tuple = (0.3, 1.0)
    row_flatten_prob: float = 0.3         # subtract per-row median (like real flattening)
    clip_percentiles: tuple = (0.5, 99.5)

    # --- label export ------------------------------------------------------- #
    approx_eps_px: float = 1.0


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #
def rotate(pts: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return pts @ np.array([[c, s], [-s, c]])


def ideal_polygon(rng, cls_name: str, cfg: Config) -> np.ndarray:
    """Return an (N,2) polygon (x, y) centred on the origin, randomly rotated."""
    if cls_name == "rectangle":
        short = rng.uniform(*cfg.rect_short_px)
        w, h = short * rng.uniform(*cfg.rect_aspect), short
        pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    elif cls_name == "triangle":
        radius = rng.uniform(*cfg.tri_radius_px)
        if rng.random() < cfg.tri_equilateral_prob:
            angles = np.deg2rad([0, 120, 240])
        else:
            while True:  # random triangle on a circle, no degenerate slivers
                gaps = rng.dirichlet([4, 4, 4]) * 2 * np.pi
                if gaps.min() > np.deg2rad(25):
                    break
            angles = np.array([0.0, gaps[0], gaps[0] + gaps[1]])
        pts = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    else:
        raise ValueError(f"Unknown class {cls_name}")
    return rotate(pts, rng.uniform(0, 2 * np.pi))


def rasterize(poly_xy: np.ndarray, size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    rr, cc = draw_polygon(poly_xy[:, 1], poly_xy[:, 0], shape=(size, size))
    mask[rr, cc] = True
    return mask


def make_broken(rng, cfg: Config, ideal: np.ndarray):
    """Remove material. Returns (mask, completeness). Falls back to the ideal mask."""
    area = ideal.sum()
    ys, xs = np.nonzero(ideal)
    eq_d = 2 * np.sqrt(area / np.pi)
    edge = ideal & ~cv2.erode(ideal.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ey, ex = np.nonzero(edge)
    if len(ey) == 0:
        return ideal, 1.0

    for _ in range(20):
        m = ideal.copy()
        kind = rng.choice(["bite", "double_bite", "cut"])
        if kind == "cut":  # slice off a cap with a random straight line
            th = rng.uniform(0, 2 * np.pi)
            proj = xs * np.cos(th) + ys * np.sin(th)
            thr = np.quantile(proj, rng.uniform(0.55, 0.92))
            rm = proj > thr
            m[ys[rm], xs[rm]] = False
        else:  # circular bite(s) centred on the boundary
            for _ in range(1 if kind == "bite" else 2):
                k = rng.integers(len(ey))
                rad = rng.uniform(0.15, 0.45) * eq_d
                rr, cc = draw_disk((ey[k], ex[k]), rad, shape=ideal.shape)
                m[rr, cc] = False

        lab = cc_label(m, connectivity=2)
        if lab.max() == 0:
            continue
        if lab.max() > 1:  # keep the largest connected piece
            sizes = np.bincount(lab.ravel())[1:]
            m = lab == (np.argmax(sizes) + 1)
        c = m.sum() / area
        if cfg.min_completeness <= c <= cfg.max_broken_completeness:
            return m, float(c)
    return ideal, 1.0


# --------------------------------------------------------------------------- #
# Scene composition
# --------------------------------------------------------------------------- #
def choose_class(rng, counts: list) -> int:
    """Mostly pick the least-represented class -> balance by instance count."""
    if rng.random() < 0.7:
        mn = min(counts)
        return int(rng.choice([i for i, c in enumerate(counts) if c == mn]))
    return int(rng.integers(len(counts)))


def sample_center(rng, cfg: Config, r: float, instances: list, touching_mode: bool):
    lo, hi = r + cfg.edge_margin, cfg.img_size - 1 - r - cfg.edge_margin
    if hi <= lo:
        return None
    if touching_mode and instances and rng.random() < cfg.anchor_prob:
        a = instances[rng.integers(len(instances))]
        d = (r + a["radius"]) * rng.uniform(*cfg.touch_dist_factor)
        ang = rng.uniform(0, 2 * np.pi)
        c = a["center"] + d * np.array([np.cos(ang), np.sin(ang)])
        if lo <= c[0] <= hi and lo <= c[1] <= hi:
            return c
    return rng.uniform(lo, hi, size=2)


def generate_scene(rng, cfg: Config, counts: list, negative: bool):
    size = cfg.img_size
    instances: list = []
    touching_mode = bool(rng.random() < cfg.touching_image_prob)
    if negative:
        return instances, touching_mode

    occ = np.zeros((size, size), dtype=bool)
    occ_dilated = occ.copy()
    kernel = np.ones((2 * cfg.separate_gap_px + 1,) * 2, np.uint8)

    n_target = int(rng.integers(cfg.n_shapes[0], cfg.n_shapes[1] + 1))
    for _ in range(n_target):
        cls = choose_class(rng, counts)
        for _attempt in range(cfg.max_place_attempts):
            poly0 = ideal_polygon(rng, CLASS_NAMES[cls], cfg)
            radius = float(np.linalg.norm(poly0, axis=1).max())
            center = sample_center(rng, cfg, radius, instances, touching_mode)
            if center is None:
                continue
            poly = poly0 + center
            ideal = rasterize(poly, size)
            area = int(ideal.sum())
            if area < cfg.min_area_px:
                continue
            if touching_mode:
                if (ideal & occ).sum() / area > cfg.max_overlap:
                    continue
            elif (ideal & occ_dilated).any():
                continue

            if rng.random() < cfg.broken_prob:
                mask, completeness = make_broken(rng, cfg, ideal)
            else:
                mask, completeness = ideal, 1.0

            instances.append(dict(
                cls=cls, poly=poly, ideal_mask=ideal, mask=mask,
                completeness=completeness, center=center, radius=radius,
                height_nm=float(rng.uniform(*cfg.shape_height_nm)),
            ))
            counts[cls] += 1
            occ |= ideal
            occ_dilated = cv2.dilate(occ.astype(np.uint8), kernel).astype(bool)
            break
    return instances, touching_mode


# --------------------------------------------------------------------------- #
# Heightmap + AFM realism
# --------------------------------------------------------------------------- #
def smooth_field(rng, size: int, sigma: float) -> np.ndarray:
    g = gaussian_filter(rng.normal(size=(size, size)), sigma, mode="reflect")
    return g / (g.std() + 1e-9)


def render_afm(rng, cfg: Config, instances: list):
    """Return (uint8 image, dict of sampled AFM parameters)."""
    size = cfg.img_size
    U = lambda rng_, r: float(rng_.uniform(*r))

    # 1. clean heightmap (nm); overlaps resolved with max()
    height = np.zeros((size, size), dtype=np.float32)
    var = 1 + cfg.in_shape_var * smooth_field(rng, size, 8)
    for inst in instances:
        h = inst["height_nm"] * var
        height = np.where(inst["mask"], np.maximum(height, h), height)

    # 2. tip convolution
    p = dict(tip_sigma=U(rng, cfg.tip_sigma_px))
    height = gaussian_filter(height, p["tip_sigma"])

    # 3. background: tilt plane + smooth random field
    yy, xx = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    tilt = U(rng, cfg.tilt_nm)
    ta, tb = rng.uniform(-1, 1, 2)
    p["tilt_nm"] = tilt
    height = height + tilt * (ta * xx + tb * yy)
    p["bg_amp"] = U(rng, cfg.smooth_bg_amp_nm)
    height = height + p["bg_amp"] * smooth_field(rng, size, U(rng, cfg.smooth_bg_sigma_px))

    # 4. white Gaussian noise
    p["noise_sigma"] = U(rng, cfg.noise_sigma_nm)
    height = height + rng.normal(0, p["noise_sigma"], size=(size, size))

    # 5. scan-line artifacts: per-row random walk + jitter + partial streaks
    step, jitter = U(rng, cfg.row_walk_step_nm), U(rng, cfg.row_jitter_nm)
    walk = np.cumsum(rng.normal(0, step, size))
    walk -= walk.mean()
    row_off = walk + rng.normal(0, jitter, size)
    height = height + row_off[:, None]
    p.update(row_walk_step=step, row_jitter=jitter, n_streaks=0)
    if rng.random() < cfg.streak_prob:
        n = int(rng.integers(cfg.streak_count[0], cfg.streak_count[1] + 1))
        p["n_streaks"] = n
        for _ in range(n):
            r0 = int(rng.integers(0, size))
            thick = int(rng.integers(1, 4))
            x0 = int(rng.integers(0, size // 2))
            amp = U(rng, cfg.streak_amp_nm) * rng.choice([-1, 1])
            height[r0:r0 + thick, x0:] += amp

    # 6. optional per-row median flattening (typical AFM post-processing)
    p["row_flattened"] = bool(rng.random() < cfg.row_flatten_prob)
    if p["row_flattened"]:
        height = height - np.median(height, axis=1, keepdims=True)

    # 7. normalise to 8-bit with percentile clipping
    lo, hi = np.percentile(height, cfg.clip_percentiles)
    img = np.clip((height - lo) / (hi - lo + 1e-9), 0, 1)
    return (img * 255).round().astype(np.uint8), p


# --------------------------------------------------------------------------- #
# Label export
# --------------------------------------------------------------------------- #
def mask_to_polygon(mask: np.ndarray, eps: float):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, eps, True)[:, 0, :].astype(float)
    return approx if len(approx) >= 3 else None


def yolo_label_lines(instances: list, cfg: Config) -> list:
    """Labels come from the FINAL (possibly broken) masks."""
    lines, s = [], cfg.img_size
    for inst in instances:
        poly = mask_to_polygon(inst["mask"], cfg.approx_eps_px)
        if poly is None:
            inst["exported"] = False
            continue
        inst["exported"] = True
        pts = np.clip(poly / s, 0, 1)
        lines.append(f"{inst['cls']} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in pts))
    return lines


# --------------------------------------------------------------------------- #
# Dataset writing
# --------------------------------------------------------------------------- #
def generate_split(split: str, n: int, split_id: int, cfg: Config, out: Path, seed: int):
    for sub in ("images", "labels", "meta"):
        (out / sub / split).mkdir(parents=True, exist_ok=True)
    counts = [0] * len(CLASS_NAMES)
    completeness_all, n_neg = [], 0

    for i in range(n):
        rng = np.random.default_rng([seed, split_id, i])   # reproducible per image
        negative = bool(rng.random() < cfg.negative_frac)
        instances, touching = generate_scene(rng, cfg, counts, negative)
        img, afm = render_afm(rng, cfg, instances)
        lines = yolo_label_lines(instances, cfg)
        n_neg += int(len(instances) == 0)

        name = f"{split}_{i:05d}"
        cv2.imwrite(str(out / "images" / split / f"{name}.png"), img)
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

        meta = dict(
            image=f"{name}.png", negative=negative, touching_mode=touching, afm=afm,
            px_nm=cfg.px_nm,
            instances=[dict(
                class_id=k["cls"], class_name=CLASS_NAMES[k["cls"]],
                completeness=round(k["completeness"], 4),
                height_nm=round(k["height_nm"], 3),
                ideal_polygon_px=np.round(k["poly"], 2).tolist(),
                exported=k["exported"],
            ) for k in instances],
        )
        (out / "meta" / split / f"{name}.json").write_text(json.dumps(meta))
        completeness_all += [k["completeness"] for k in instances]

        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"  [{split}] {i + 1}/{n}", end="\r", flush=True)

    tot = max(sum(counts), 1)
    comp = np.array(completeness_all) if completeness_all else np.array([1.0])
    print(f"\n[{split}] images={n}  empty/negative={n_neg} ({100 * n_neg / max(n, 1):.1f}%)")
    for name, c in zip(CLASS_NAMES, counts):
        print(f"   {name:<10} {c:>6} instances ({100 * c / tot:.1f}%)")
    print(f"   broken shapes (completeness<1): {(comp < 1).mean() * 100:.1f}%  "
          f"mean completeness={comp.mean():.3f}")


def write_yaml(out: Path):
    lines = [f"path: {out.resolve()}", "train: images/train", "val: images/val", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    (out / "data.yaml").write_text("\n".join(lines) + "\n")


def write_previews(out: Path, split: str, n: int):
    """Overlay the *exported label files* on the images (catches x/y flips etc.)."""
    pdir = out / "preview"
    pdir.mkdir(exist_ok=True)
    colors = [(0, 200, 255), (255, 120, 0), (0, 220, 0), (200, 0, 200)]  # BGR
    for p in sorted((out / "images" / split).glob("*.png"))[:n]:
        img = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        h, w = img.shape[:2]
        meta = json.loads((out / "meta" / split / f"{p.stem}.json").read_text())
        lab = (out / "labels" / split / f"{p.stem}.txt").read_text().splitlines()
        exported = [m for m in meta["instances"] if m["exported"]]
        for ln, m in zip(lab, exported):
            v = ln.split()
            cls = int(v[0])
            pts = (np.array(v[1:], float).reshape(-1, 2) * [w, h]).round().astype(np.int32)
            cv2.polylines(img, [pts], True, colors[cls % len(colors)], 2)
            cv2.putText(img, f"{CLASS_NAMES[cls][:3]} {m['completeness']:.2f}",
                        tuple(pts.min(axis=0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colors[cls % len(colors)], 1, cv2.LINE_AA)
        cv2.imwrite(str(pdir / f"{p.stem}_overlay.png"), img)
    print(f"Wrote overlays to {pdir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("dataset"))
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--preview", type=int, default=20, help="number of train overlays to write (0=off)")
    args = ap.parse_args()

    cfg = Config(img_size=args.img_size)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Generating into {args.out.resolve()} (seed={args.seed})")
    generate_split("train", args.n_train, 0, cfg, args.out, args.seed)
    generate_split("val", args.n_val, 1, cfg, args.out, args.seed)
    write_yaml(args.out)
    if args.preview > 0:
        write_previews(args.out, "train", args.preview)
    print("Done. Train with e.g.:  yolo segment train data=%s/data.yaml model=yolo11n-seg.pt imgsz=512 epochs=50"
          % args.out)


if __name__ == "__main__":
    main()