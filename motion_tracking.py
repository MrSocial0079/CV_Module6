# CSc 8830 - Computer Vision, Assignment 6 - Part 2
# Author: Yasaswi Kompella
# Lucas-Kanade tracking from first principles + bilinear interpolation.
# Validates our custom tracker vs OpenCV on two consecutive frames per video.
# Usage: python motion_tracking.py
# Outputs saved to output/tracking/

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from typing import List, Tuple

VIDEO_PATHS = {
    "video1": "videos/video1.mp4",
    "video2": "videos/video2.mp4",
}
OUTPUT_DIR = "output/tracking"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATCH_HALF        = 7     # LK window half-width -> 15x15 patch
LK_ITERATIONS     = 20
LK_EPSILON        = 0.01
LK_PYRAMID_LEVELS = 3
PROCESS_WIDTH     = 720   # downscale before processing to keep motions small

LK_PARAMS_CV = dict(
    winSize  = (2*PATCH_HALF+1, 2*PATCH_HALF+1),
    maxLevel = LK_PYRAMID_LEVELS - 1,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                LK_ITERATIONS, LK_EPSILON),
)

FEATURE_PARAMS = dict(
    maxCorners   = 30,
    qualityLevel = 0.01,
    minDistance  = 20,
    blockSize    = 7,
)


# --- Bilinear interpolation (from scratch) ---
# Given sub-pixel (x, y), interpolate from the four surrounding integer pixels:
#   Q11=(x0,y0)  Q21=(x0+1,y0)  Q12=(x0,y0+1)  Q22=(x0+1,y0+1)
# R1 = (1-dx)*Q11 + dx*Q21        <- interpolate top row in x
# R2 = (1-dx)*Q12 + dx*Q22        <- interpolate bottom row in x
# I  = (1-dy)*R1  + dy*R2         <- interpolate between rows in y
# => I = (1-dx)(1-dy)*Q11 + dx(1-dy)*Q21 + (1-dx)dy*Q12 + dx*dy*Q22
def bilinear_interpolate(img, x, y):
    h, w = img.shape
    x = max(0.0, min(float(x), w - 1.0001))
    y = max(0.0, min(float(y), h - 1.0001))
    x0, y0 = int(x), int(y)
    dx, dy = x - x0, y - y0
    Q11 = float(img[y0,             x0])
    Q21 = float(img[y0,             min(x0+1, w-1)])
    Q12 = float(img[min(y0+1, h-1), x0])
    Q22 = float(img[min(y0+1, h-1), min(x0+1, w-1)])
    R1 = (1 - dx) * Q11 + dx * Q21
    R2 = (1 - dx) * Q12 + dx * Q22
    return (1 - dy) * R1 + dy * R2


def bilinear_patch(img, cx, cy, half):
    # extract a (2*half+1) x (2*half+1) patch centred at sub-pixel (cx, cy)
    size  = 2 * half + 1
    patch = np.zeros((size, size), dtype=np.float32)
    for j in range(size):
        for i in range(size):
            patch[j, i] = bilinear_interpolate(img, cx + (i - half), cy + (j - half))
    return patch


# --- Spatial gradients via Sobel operator ---
def image_gradients(img):
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return Ix, Iy


# --- Single-level Lucas-Kanade ---
# Goal: find (u,v) minimising E(u,v) = sum_W [ I2(x+u,y+v) - I1(x,y) ]^2
# Brightness constancy: I1(x,y) = I2(x+u,y+v)
# Taylor expansion:     I2(x+u,y+v) ~ I2(x,y) + Ix*u + Iy*v
# Substituting:         Ix*u + Iy*v + It = 0   (OFCE, one eq / two unknowns)
# Over patch W (N pixels):  A*d = -b
#   A = [[Ix1,Iy1],[Ix2,Iy2],...]   b = [It1, It2, ...]   d = [u,v]^T
# Least-squares normal equations:  (A^T A) d = -A^T b  =>  d = -(A^T A)^-1 A^T b
# Structure tensor M = A^T A = [[SIx2, SIxIy],[SIxIy, SIy2]]
# Iterate: warp I2 by current d, compute residual, update until convergence.
def lk_single_level(I1, I2, pt, half=PATCH_HALF,
                    n_iter=LK_ITERATIONS, epsilon=LK_EPSILON):
    cx, cy = float(pt[0]), float(pt[1])
    h, w   = I1.shape
    Ix, Iy = image_gradients(I1)
    u, v   = 0.0, 0.0

    for _ in range(n_iter):
        AtA = np.zeros((2, 2), dtype=np.float64)  # structure tensor M = A^T A
        Atb = np.zeros(2,      dtype=np.float64)  # A^T b

        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                px1, py1 = cx + dx, cy + dy
                if px1 < 1 or px1 >= w-1 or py1 < 1 or py1 >= h-1:
                    continue

                ix = bilinear_interpolate(Ix, px1, py1)
                iy = bilinear_interpolate(Iy, px1, py1)
                i1 = bilinear_interpolate(I1, px1, py1)
                i2 = bilinear_interpolate(I2, px1 + u, py1 + v)
                it = i2 - i1  # temporal difference

                AtA[0, 0] += ix * ix  # SIx2
                AtA[0, 1] += ix * iy  # SIxIy
                AtA[1, 0] += ix * iy
                AtA[1, 1] += iy * iy  # SIy2
                Atb[0]    += ix * it  # SIx*It
                Atb[1]    += iy * it  # SIy*It

        # M must be invertible (both eigenvalues large = good corner)
        if np.linalg.eigvalsh(AtA).min() < 1e-4:
            return cx, cy, False

        # solve d = -M^-1 A^T b
        delta = np.linalg.solve(AtA, -Atb)
        u += delta[0]
        v += delta[1]

        if np.sqrt(delta[0]**2 + delta[1]**2) < epsilon:
            break

    return cx + u, cy + v, True


# --- Coarse-to-fine (pyramidal) LK ---
# Build Gaussian pyramids; track from coarsest to finest level.
# Displacement from coarser level is scaled up and used as initial guess.
def lk_pyramidal(I1_orig, I2_orig, points, levels=LK_PYRAMID_LEVELS):
    pyr1 = [I1_orig]
    pyr2 = [I2_orig]
    for _ in range(levels - 1):
        pyr1.append(cv2.pyrDown(pyr1[-1]))
        pyr2.append(cv2.pyrDown(pyr2[-1]))

    N             = len(points)
    new_points    = points.copy().astype(np.float64)
    statuses      = np.ones(N, dtype=bool)
    displacements = np.zeros((N, 2), dtype=np.float64)

    for level in range(levels - 1, -1, -1):
        scale = 2.0 ** level
        I1_l  = pyr1[level].astype(np.float32)
        I2_l  = pyr2[level].astype(np.float32)

        for i, pt in enumerate(points):
            if not statuses[i]:
                continue
            cx_l   = pt[0] / scale
            cy_l   = pt[1] / scale
            # propagate displacement estimate from coarser level
            init_x = cx_l + displacements[i, 0]
            init_y = cy_l + displacements[i, 1]

            x2, y2, ok = lk_single_level(I1_l, I2_l, (init_x, init_y),
                                          half=max(2, PATCH_HALF // (level + 1)))
            statuses[i]         = ok
            displacements[i, 0] = x2 - cx_l
            displacements[i, 1] = y2 - cy_l

            if level == 0:
                new_points[i, 0] = x2
                new_points[i, 1] = y2

        if level > 0:
            displacements *= 2.0  # scale up for next finer level

    return new_points, statuses


def validate_tracking(video_name, video_path, report_lines):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    start = int(10 * fps)  # sample at 10s mark
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()
    cap.release()

    if not ret1 or not ret2:
        print(f"[ERROR] Could not read frames from {video_path}")
        return

    def _resize(f):
        h, w = f.shape[:2]
        if PROCESS_WIDTH and w > PROCESS_WIDTH:
            s = PROCESS_WIDTH / w
            return cv2.resize(f, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        return f

    frame1 = _resize(frame1)
    frame2 = _resize(frame2)
    h_proc, w_proc = frame1.shape[:2]
    print(f"  Processing at {w_proc}x{h_proc}")

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    pts_raw = cv2.goodFeaturesToTrack(gray1.astype(np.uint8), **FEATURE_PARAMS)
    if pts_raw is None or len(pts_raw) == 0:
        print(f"[WARN] No features found in {video_name}")
        return

    pts_in = pts_raw[:, 0, :]
    N      = len(pts_in)

    # our custom LK
    pts_our, status_our = lk_pyramidal(gray1, gray2, pts_in)

    # OpenCV LK for comparison
    pts_cv_raw, status_cv, _ = cv2.calcOpticalFlowPyrLK(
        gray1.astype(np.uint8), gray2.astype(np.uint8),
        pts_raw, None, **LK_PARAMS_CV
    )
    pts_cv2   = pts_cv_raw[:, 0, :]
    status_cv = status_cv[:, 0]

    report_lines.append(f"\n{'='*70}")
    report_lines.append(f"  VIDEO: {video_name}  ({video_path})")
    report_lines.append(f"  Frames: {start} & {start+1}  (t={start/fps:.2f}s)")
    report_lines.append(f"{'='*70}")
    report_lines.append(
        f"  {'Pt':>3}  {'x1':>6} {'y1':>6}  "
        f"{'x_ours':>8} {'y_ours':>8}  "
        f"{'x_cv2':>8} {'y_cv2':>8}  "
        f"{'|Dx|':>6} {'|Dy|':>6}  {'OK?':>5}"
    )
    report_lines.append("  " + "-"*68)

    errors = []
    for i in range(N):
        if not (status_our[i] and status_cv[i]):
            continue
        x1, y1  = pts_in[i]
        xo, yo  = pts_our[i]
        xc, yc  = pts_cv2[i]
        ex, ey  = abs(xo - xc), abs(yo - yc)
        err     = np.sqrt(ex**2 + ey**2)
        errors.append(err)
        report_lines.append(
            f"  {i:>3}  {x1:>6.1f} {y1:>6.1f}  "
            f"{xo:>8.3f} {yo:>8.3f}  "
            f"{xc:>8.3f} {yc:>8.3f}  "
            f"{ex:>6.3f} {ey:>6.3f}  {'ok' if err < 1.5 else '--':>5}"
        )

    if errors:
        report_lines.append("  " + "-"*68)
        report_lines.append(
            f"  Mean error: {np.mean(errors):.4f} px   Max: {np.max(errors):.4f} px"
        )
        report_lines.append(
            f"  Within 1.5px: {sum(e < 1.5 for e in errors)}/{len(errors)}"
        )

    # bilinear interpolation spot-check vs cv2.remap
    report_lines.append(f"\n  --- Bilinear Interpolation Validation ---")
    report_lines.append(
        f"  {'Pixel (x,y)':>14}  {'Sub-pixel':>10}  "
        f"{'Our bilinear':>14}  {'cv2.remap':>10}  {'diff':>8}"
    )
    for i in range(min(5, N)):
        xf, yf  = pts_in[i, 0] + 0.37, pts_in[i, 1] + 0.82
        our_val = bilinear_interpolate(gray2, xf, yf)
        cv_val  = float(cv2.remap(gray2,
                                  np.array([[xf]], dtype=np.float32),
                                  np.array([[yf]], dtype=np.float32),
                                  cv2.INTER_LINEAR)[0, 0])
        report_lines.append(
            f"  ({int(xf):>5},{int(yf):>5})  ({xf:.2f},{yf:.2f})  "
            f"{our_val:>14.4f}  {cv_val:>10.4f}  {abs(our_val-cv_val):>8.5f}"
        )

    _save_tracking_figure(video_name, frame1, frame2,
                          pts_in, pts_our, pts_cv2, status_our, status_cv)
    print(f"  Mean error: {np.mean(errors):.3f} px" if errors else "  Done")


def _save_tracking_figure(name, frame1, frame2,
                           pts_in, pts_our, pts_cv2, status_our, status_cv):
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(f"{name} — Motion Tracking: Our LK vs OpenCV LK",
                 fontsize=13, fontweight="bold")

    # (a) seed points in frame 1
    axes[0].imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    axes[0].scatter(pts_in[:, 0], pts_in[:, 1], s=40, c="yellow",
                    marker="o", linewidths=1, edgecolors="black", label="Seeds")
    axes[0].set_title("(a) Frame t — Feature Seeds"); axes[0].axis("off")
    axes[0].legend(fontsize=8, loc="upper right")

    # (b) our LK predicted positions
    axes[1].imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    for i, (pt, ok) in enumerate(zip(pts_our, status_our)):
        if ok:
            axes[1].plot(pt[0], pt[1], "bo", ms=6, mec="white", mew=0.8)
            axes[1].annotate("", xytext=(pts_in[i, 0], pts_in[i, 1]),
                             xy=(pt[0], pt[1]),
                             arrowprops=dict(arrowstyle="->", color="cyan", lw=1.2))
    axes[1].set_title("(b) Our LK Predictions"); axes[1].axis("off")

    # (c) OpenCV LK predicted positions
    axes[2].imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    for i, (pt, ok) in enumerate(zip(pts_cv2, status_cv)):
        if ok:
            axes[2].plot(pt[0], pt[1], "r+", ms=10, mew=2)
            axes[2].annotate("", xytext=(pts_in[i, 0], pts_in[i, 1]),
                             xy=(pt[0], pt[1]),
                             arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
    axes[2].set_title("(c) OpenCV LK Predictions"); axes[2].axis("off")

    # (d) overlay — green lines show the error between the two
    axes[3].imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    for i in range(len(pts_in)):
        if status_our[i]:
            axes[3].plot(pts_our[i, 0], pts_our[i, 1], "bo", ms=7, mec="white", mew=0.7)
        if status_cv[i]:
            axes[3].plot(pts_cv2[i, 0], pts_cv2[i, 1], "r+", ms=9, mew=2)
        if status_our[i] and status_cv[i]:
            axes[3].plot([pts_our[i, 0], pts_cv2[i, 0]],
                         [pts_our[i, 1], pts_cv2[i, 1]],
                         "g-", lw=1.5, alpha=0.7)
    axes[3].legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=8, label="Our LK"),
        Line2D([0], [0], marker="+", color="red", markersize=10, markeredgewidth=2, label="OpenCV LK"),
        Line2D([0], [0], color="green", lw=1.5, label="Error"),
    ], fontsize=8, loc="upper right")
    axes[3].set_title("(d) Overlay — green lines = error"); axes[3].axis("off")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{name}_tracking_validation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure  : {out}")


def demo_bilinear_interpolation():
    # Compare nearest-neighbour vs our bilinear on a known 5x5 patch,
    # and annotate the formula weights for one sub-pixel query point.
    src = np.array([
        [ 10,  50, 100,  80,  40],
        [ 30, 120, 200, 160,  70],
        [ 60, 180, 255, 190, 100],
        [ 40, 130, 190, 150,  60],
        [ 20,  60,  90,  70,  30],
    ], dtype=np.float32)

    H, W  = src.shape
    SCALE = 10

    nn_up = cv2.resize(src, (W * SCALE, H * SCALE),
                       interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    bil_up = np.zeros((H * SCALE, W * SCALE), dtype=np.float32)
    for r in range(H * SCALE):
        for c in range(W * SCALE):
            xs = c / SCALE * (W - 1) / (W - 1 / SCALE)
            ys = r / SCALE * (H - 1) / (H - 1 / SCALE)
            bil_up[r, c] = bilinear_interpolate(src, xs, ys)
    bil_up = np.clip(bil_up, 0, 255).astype(np.uint8)

    qx, qy = 1.7, 2.3
    x0, y0 = int(qx), int(qy)
    dx, dy = qx - x0, qy - y0
    Q11, Q21 = src[y0, x0], src[y0, x0+1]
    Q12, Q22 = src[y0+1, x0], src[y0+1, x0+1]
    result = (1-dx)*(1-dy)*Q11 + dx*(1-dy)*Q21 + (1-dx)*dy*Q12 + dx*dy*Q22

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Bilinear Interpolation — Derivation & Demonstration",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.imshow(src, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    for r in range(H):
        for c in range(W):
            ax.text(c, r, str(int(src[r, c])), ha="center", va="center",
                    color="lime" if src[r, c] < 128 else "black", fontsize=10)
    ax.set_title("(a) Source 5x5 patch", fontsize=10)
    ax.set_xticks(range(W)); ax.set_yticks(range(H))
    ax.grid(color="white", linewidth=1)

    axes[1].imshow(nn_up, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    axes[1].set_title("(b) Nearest-Neighbour x10\n(blocky)", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(bil_up, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    axes[2].set_title("(c) Our Bilinear x10\n(smooth)", fontsize=10)
    axes[2].axis("off")

    ax = axes[3]
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal"); ax.invert_yaxis()
    for cx, cy_, val, lbl in [(0,0,Q11,"Q11"),(1,0,Q21,"Q21"),
                               (0,1,Q12,"Q12"),(1,1,Q22,"Q22")]:
        ax.plot(cx, cy_, "ks", ms=18, zorder=3)
        ax.text(cx, cy_, f"{lbl}\n={int(val)}", ha="center", va="center",
                color="white", fontsize=9, zorder=4)
    ax.plot(dx, dy, "r*", ms=16, zorder=5, label=f"Query ({qx},{qy})")
    for cx, cy_, wtxt in [(0,0,f"w={(1-dx)*(1-dy):.2f}"), (1,0,f"w={dx*(1-dy):.2f}"),
                           (0,1,f"w={(1-dx)*dy:.2f}"),    (1,1,f"w={dx*dy:.2f}")]:
        ax.text(cx+0.07, cy_+0.15, wtxt, ha="center", fontsize=8, color="red")
    ax.set_title(f"(d) Formula Weights\nQuery=({qx},{qy}) -> {result:.1f}", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("x ->"); ax.set_ylabel("y down")
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "bilinear_interp_demo.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bilinear demo: {out}")


def main():
    report_lines = [
        "=" * 70,
        "  CSc 8830 Assignment 6 - Part 2: Motion Tracking Validation Report",
        "=" * 70,
        "",
        "  LK minimises SSD over patch W:  E(u,v) = sum[ I2(x+u,y+v) - I1(x,y) ]^2",
        "  Brightness constancy + Taylor:  Ix*u + Iy*v + It = 0  (OFCE)",
        "  Matrix form over patch:         A*d = -b",
        "  Least squares solution:         d = -(A^T A)^-1 A^T b",
        "  Structure tensor M = A^T A = [[SIx2, SIxIy], [SIxIy, SIy2]]",
        "",
        "  Bilinear: I(x+dx,y+dy) = (1-dx)(1-dy)*Q11 + dx(1-dy)*Q21",
        "                          + (1-dx)dy*Q12   + dx*dy*Q22",
        "",
    ]

    print("\n[Bilinear demo]")
    demo_bilinear_interpolation()

    for name, path in VIDEO_PATHS.items():
        if not os.path.isfile(path):
            print(f"[SKIP] {path} not found")
            continue
        print(f"\n[Tracking] {name}  ({path})")
        validate_tracking(name, path, report_lines)

    report_path = os.path.join(OUTPUT_DIR, "tracking_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n[Done] Report : {report_path}")
    print(f"[Done] Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
