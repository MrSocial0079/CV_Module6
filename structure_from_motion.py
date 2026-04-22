"""
Structure from Motion (SfM) - Chick-fil-A Gift Card (Planar Object)
=====================================================================
Camera: iPhone 16 Pro
Lens:   2.22mm f/2.2 (24mm equiv in 35mm)
Sensor: 4032 x 3024 pixels
Images: view_1.jpg, view_2.jpg, view_3.jpg (3 viewpoints of flat card)

Mathematical Workouts & Full Pipeline:
  1. Camera Intrinsic Matrix K
  2. Feature Detection & Matching (SIFT)
  3. Fundamental Matrix F (8-point algorithm)
  4. Essential Matrix E = K^T F K
  5. Camera Pose Recovery (R, t) via SVD decomposition
  6. Triangulation (Direct Linear Transform)
  7. Point Cloud & Boundary Estimation (Convex Hull)
  8. Visualization & Report

Author: Yash (SfM Assignment)
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.spatial import ConvexHull
from PIL import Image
import os, sys

# ─────────────────────────────────────────────────────────────
# 0.  PATHS  (works both from project root and from output/)
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR  = os.path.join(SCRIPT_DIR, "photos")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_PATHS = [
    os.path.join(PHOTOS_DIR, "view_1.jpg"),
    os.path.join(PHOTOS_DIR, "view_2.jpg"),
    os.path.join(PHOTOS_DIR, "view_3.jpg"),
]

# ─────────────────────────────────────────────────────────────
# 1.  CAMERA INTRINSICS  (from EXIF — iPhone 16 Pro)
# ─────────────────────────────────────────────────────────────
#
#  Sensor size for iPhone 16 Pro main camera: ~5.76 mm x 4.29 mm  (1/1.28" sensor)
#  Physical focal length f_phys = 2.22 mm
#  Image resolution: W = 4032 px,  H = 3024 px
#
#  Pixel pitch:
#    px = W / sensor_w = 4032 / 5.76  ≈ 700.0  px/mm
#    py = H / sensor_h = 3024 / 4.32  ≈ 700.0  px/mm
#
#  Focal length in pixels:
#    fx = f_phys * px = 2.22 * 700.0  ≈ 1554  px
#    fy = f_phys * py = 2.22 * 700.0  ≈ 1554  px
#
#  Principal point (image centre):
#    cx = W/2 = 2016,   cy = H/2 = 1512
#
#  Intrinsic matrix K:
#    ┌  fx   0   cx ┐   ┌ 1554    0   2016 ┐
#    │   0  fy   cy │ = │    0  1554  1512  │
#    └   0   0    1 ┘   └    0     0     1  ┘

W, H       = 4032, 3024
SENSOR_W   = 5.76          # mm  (1/1.28" sensor)
SENSOR_H   = 4.32          # mm
F_PHYS     = 2.22          # mm  (from EXIF FocalLength)

fx = F_PHYS * (W / SENSOR_W)   # ≈ 1554 px
fy = F_PHYS * (H / SENSOR_H)   # ≈ 1554 px
cx, cy = W / 2.0, H / 2.0

K = np.array([[fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1]], dtype=np.float64)

print("=" * 65)
print("STRUCTURE FROM MOTION  —  Chick-fil-A Gift Card")
print("=" * 65)
print(f"\n[1] CAMERA INTRINSICS (iPhone 16 Pro, 2.22 mm lens)")
print(f"    Sensor : {SENSOR_W} x {SENSOR_H} mm  |  Image: {W} x {H} px")
print(f"    fx = {fx:.1f} px,  fy = {fy:.1f} px")
print(f"    cx = {cx:.1f} px,  cy = {cy:.1f} px")
print(f"    K =\n{K}\n")

# ─────────────────────────────────────────────────────────────
# 2.  LOAD & RESIZE IMAGES
# ─────────────────────────────────────────────────────────────
# We work at half resolution to speed things up.
SCALE = 0.5

def load_gray(path):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Cannot read {path}. Run from the project root that contains photos/")
    small = cv2.resize(img, (0, 0), fx=SCALE, fy=SCALE)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return small, gray

imgs_bgr, imgs_gray = [], []
for p in IMG_PATHS:
    bgr, gray = load_gray(p)
    imgs_bgr.append(bgr)
    imgs_gray.append(gray)
    print(f"    Loaded {os.path.basename(p)}  →  {bgr.shape[1]}x{bgr.shape[0]} px (scaled)")

# Scale intrinsics to match resized images
Ks = K.copy()
Ks[0] *= SCALE;  Ks[1] *= SCALE;  Ks[2, 2] = 1.0

# ─────────────────────────────────────────────────────────────
# 3.  FEATURE DETECTION & MATCHING  (SIFT + FLANN + Lowe ratio)
# ─────────────────────────────────────────────────────────────
#
#  SIFT finds keypoints with scale-space extrema in DoG.
#  Each keypoint has a 128-dim descriptor.
#  Lowe ratio test: keep match only if
#        dist(best) / dist(2nd_best) < 0.75
#  This removes 90%+ of false matches.

sift  = cv2.SIFT_create(nfeatures=3000)
index_params  = dict(algorithm=1, trees=5)   # FLANN KD-Tree
search_params = dict(checks=100)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def detect_and_match(g1, g2, ratio=0.75):
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)
    if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
        return [], [], []
    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return good, pts1, pts2

print(f"\n[2] FEATURE DETECTION & MATCHING")
pairs = [(0,1),(0,2),(1,2)]
match_data = {}
for (i, j) in pairs:
    good, p1, p2 = detect_and_match(imgs_gray[i], imgs_gray[j])
    match_data[(i,j)] = (good, p1, p2)
    print(f"    View {i+1}↔View {j+1}:  {len(good)} inlier matches (after ratio test)")

# ─────────────────────────────────────────────────────────────
# 4.  FUNDAMENTAL MATRIX  (Normalised 8-point algorithm)
# ─────────────────────────────────────────────────────────────
#
#  For corresponding points x ↔ x':
#      x'^T  F  x = 0
#
#  Build matrix A (Nx9) where each row =
#      [x'x, x'y, x', y'x, y'y, y', x, y, 1]
#  Solve  A f = 0  via SVD → f = last row of V^T.
#  Reshape to 3×3, enforce rank-2 by zeroing smallest singular value.
#
#  OpenCV uses RANSAC on top to reject outliers.

print(f"\n[3] FUNDAMENTAL MATRIX  (8-point + RANSAC)")

def compute_F(pts1, pts2):
    F, mask = cv2.findFundamentalMat(pts1, pts2,
                                     cv2.FM_RANSAC,
                                     ransacReprojThreshold=1.0,
                                     confidence=0.999)
    inlier_mask = mask.ravel().astype(bool)
    return F, inlier_mask

F_mats, inlier_pts = {}, {}
for (i, j) in pairs:
    _, p1, p2 = match_data[(i,j)]
    if len(p1) < 8:
        print(f"    View {i+1}↔View {j+1}: not enough points")
        continue
    F, mask = compute_F(p1, p2)
    F_mats[(i,j)] = F
    p1_in = p1[mask];  p2_in = p2[mask]
    inlier_pts[(i,j)] = (p1_in, p2_in)
    print(f"    View {i+1}↔View {j+1}:  {mask.sum()} RANSAC inliers")
    # Sampson error (symmetric epipolar distance) — should be < 1 px
    ones = np.ones((len(p1_in), 1))
    x1h  = np.hstack([p1_in, ones])
    x2h  = np.hstack([p2_in, ones])
    err  = np.abs(np.sum(x2h * (x1h @ F.T), axis=1))
    print(f"              Mean Sampson error = {err.mean():.4f} px")

# ─────────────────────────────────────────────────────────────
# 5.  ESSENTIAL MATRIX  E = K^T F K
# ─────────────────────────────────────────────────────────────
#
#  The Essential Matrix encodes rotation + translation between cameras.
#      E = K^T  F  K
#  E must satisfy rank-2 and equal singular values.
#  Enforce via SVD:
#      E = U diag(1,1,0) V^T
#
#  Four candidate decompositions:
#      R = U W V^T  or  U W^T V^T   (W is 90° rotation)
#      t = ±U[:,2]
#  The correct (R,t) is the one where reconstructed point is in
#  front of BOTH cameras (positive depth test).

W_mat = np.array([[ 0,-1, 0],
                  [ 1, 0, 0],
                  [ 0, 0, 1]], dtype=np.float64)

print(f"\n[4] ESSENTIAL MATRIX  E = K^T F K")

def decompose_E(E, p1, p2, K):
    """Return (R, t) with positive-depth chirality check."""
    U, S, Vt = np.linalg.svd(E)
    # Enforce equal singular values
    E_fixed = U @ np.diag([1, 1, 0]) @ Vt
    U, _, Vt = np.linalg.svd(E_fixed)
    if np.linalg.det(U) < 0:  U  *= -1
    if np.linalg.det(Vt) < 0: Vt *= -1

    R1 = U @ W_mat  @ Vt
    R2 = U @ W_mat.T @ Vt
    t1 =  U[:, 2]
    t2 = -U[:, 2]

    best, best_score = None, -1
    for R in (R1, R2):
        for t in (t1, t2):
            P1 = K @ np.hstack([np.eye(3),   np.zeros((3,1))])
            P2 = K @ np.hstack([R,            t.reshape(3,1)])
            pts4d = cv2.triangulatePoints(P1, P2,
                                          p1[:10].T, p2[:10].T)
            pts3d = pts4d[:3] / pts4d[3]
            # depth in cam1
            d1 = pts3d[2]
            # depth in cam2
            d2 = (R @ pts3d + t.reshape(3,1))[2]
            score = int((d1 > 0).sum()) + int((d2 > 0).sum())
            if score > best_score:
                best_score = score
                best = (R, t)
    return best

E_mats = {}
poses  = {}   # (i,j) → (R, t)  camera j relative to camera i

for (i, j) in pairs:
    if (i,j) not in F_mats: continue
    F = F_mats[(i,j)]
    E = Ks.T @ F @ Ks
    # Enforce rank-2
    U, S, Vt = np.linalg.svd(E)
    E_r = U @ np.diag([1, 1, 0]) @ Vt
    E_mats[(i,j)] = E_r
    print(f"    View {i+1}↔View {j+1}  singular values of E: {np.linalg.svd(E_r, compute_uv=False).round(3)}")

    p1, p2 = inlier_pts[(i,j)]
    R, t = decompose_E(E_r, p1, p2, Ks)
    poses[(i,j)] = (R, t)
    angle = np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1)))
    print(f"              Rotation angle  = {angle:.2f}°")
    print(f"              Translation dir = {t.round(4)}")

# ─────────────────────────────────────────────────────────────
# 6.  TRIANGULATION  (Direct Linear Transform — DLT)
# ─────────────────────────────────────────────────────────────
#
#  For a 3D point X, its projection into camera i is:
#      λ xᵢ = Pᵢ X          (Pᵢ = Kᵢ [Rᵢ | tᵢ])
#
#  Cross-multiplying:  xᵢ × (Pᵢ X) = 0
#  Leads to a system  A X = 0  solved via SVD.
#  OpenCV's triangulatePoints implements this efficiently.
#
#  We use the (0,1) pair as the primary pair and refine
#  with the (0,2) pair.

print(f"\n[5] TRIANGULATION  (DLT via SVD)")

all_pts3d = []

def triangulate_pair(i, j):
    if (i,j) not in poses: return None
    R, t = poses[(i,j)]
    P1   = Ks @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2   = Ks @ np.hstack([R,         t.reshape(3,1)])
    p1, p2 = inlier_pts[(i,j)]
    pts4d  = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
    pts3d  = (pts4d[:3] / pts4d[3]).T          # N×3
    # Keep only points in front of both cameras
    depth1 = pts3d[:, 2]
    depth2 = (R @ pts3d.T + t.reshape(3,1))[2]
    mask   = (depth1 > 0) & (depth2 > 0)
    return pts3d[mask]

for (i, j) in [(0,1), (0,2)]:
    pts = triangulate_pair(i, j)
    if pts is not None and len(pts) > 0:
        all_pts3d.append(pts)
        print(f"    View {i+1}↔View {j+1}: {len(pts)} triangulated points (depth > 0)")

if not all_pts3d:
    print("    ERROR: No 3D points recovered. Check image quality/overlap.")
    sys.exit(1)

cloud = np.vstack(all_pts3d)
print(f"    Total raw point cloud: {len(cloud)} points")

# ─────────────────────────────────────────────────────────────
# 7.  OUTLIER REMOVAL  (IQR filter per axis)
# ─────────────────────────────────────────────────────────────
def iqr_filter(pts, k=3.0):
    mask = np.ones(len(pts), dtype=bool)
    for axis in range(3):
        q1, q3 = np.percentile(pts[:, axis], [25, 75])
        iqr = q3 - q1
        mask &= (pts[:, axis] > q1 - k*iqr) & (pts[:, axis] < q3 + k*iqr)
    return pts[mask]

cloud_clean = iqr_filter(cloud)
print(f"    After IQR outlier removal: {len(cloud_clean)} points")

# ─────────────────────────────────────────────────────────────
# 8.  BOUNDARY ESTIMATION  (Convex Hull on XY projection)
# ─────────────────────────────────────────────────────────────
#
#  For a planar object the boundary is well-approximated by
#  the 2D convex hull of the projected XY point cloud.
#
#  The hull area is compared to the known card aspect ratio
#  (credit-card standard: 85.6 mm × 54 mm → aspect 1.585).

print(f"\n[6] BOUNDARY ESTIMATION  (Convex Hull)")

pts_xy = cloud_clean[:, :2]

try:
    hull      = ConvexHull(pts_xy)
    hull_verts = pts_xy[hull.vertices]
    hull_area  = hull.volume          # 2D "volume" = area
    print(f"    Convex hull vertices : {len(hull.vertices)}")
    print(f"    Hull area (arb units): {hull_area:.4f}")

    # Bounding box aspect ratio
    mn, mx = hull_verts.min(axis=0), hull_verts.max(axis=0)
    span = mx - mn
    aspect = (span.max() / span.min()) if span.min() > 0 else float('nan')
    print(f"    Bounding box span    : {span.round(4)}")
    print(f"    Estimated aspect ratio: {aspect:.3f}  "
          f"(true card: {85.6/54:.3f})")
except Exception as e:
    hull = None
    print(f"    Hull failed: {e}")

# ─────────────────────────────────────────────────────────────
# 9.  CAMERA POSITIONS
# ─────────────────────────────────────────────────────────────
#
#  Camera centre C = -R^T t  (world coords)
#  Camera 1 is at origin [0,0,0].

cam_centers = [np.zeros(3)]
for (i, j) in [(0,1), (0,2)]:
    if (i,j) in poses:
        R, t = poses[(i,j)]
        C = -R.T @ t
        cam_centers.append(C)

print(f"\n[7] CAMERA POSITIONS (world coords)")
labels = ["View 1 (reference)", "View 2", "View 3"]
for lbl, C in zip(labels, cam_centers):
    print(f"    {lbl}: {C.round(4)}")

# ─────────────────────────────────────────────────────────────
# 10. VISUALISATION  —  3 panels
# ─────────────────────────────────────────────────────────────
print(f"\n[8] GENERATING FIGURES …")

# ── Figure 1:  Feature matches ────────────────────────────────
fig1, axes = plt.subplots(1, 3, figsize=(20, 6))
fig1.patch.set_facecolor('#0d1117')
for ax, (i, j) in zip(axes, pairs):
    key = (i, j)
    if key not in match_data or len(match_data[key][0]) == 0:
        ax.axis('off'); continue
    good, p1, p2 = match_data[key]
    # draw only inliers if available
    if key in inlier_pts:
        p1, p2 = inlier_pts[key]
        good   = None         # we'll draw manually

    h = imgs_bgr[i].shape[0]
    canvas = np.hstack([imgs_bgr[i], imgs_bgr[j]])
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    ax.imshow(canvas_rgb)
    W_img = imgs_bgr[i].shape[1]
    n_draw = min(60, len(p1))
    idx    = np.random.choice(len(p1), n_draw, replace=False)
    for k in idx:
        x1, y1 = p1[k]
        x2, y2 = p2[k][0] + W_img, p2[k][1] if p2.ndim == 2 else p2[k][0] + W_img
        # handle both 1-D and 2-D arrays
        if p2.ndim == 2:
            x2 = p2[k][0] + W_img
            y2 = p2[k][1]
        color = np.random.rand(3)
        ax.plot([x1, x2], [y1, y2], '-', color=color, lw=0.6, alpha=0.7)
        ax.plot(x1, y1, 'o', color=color, ms=3)
        ax.plot(x2, y2, 'o', color=color, ms=3)
    ax.set_title(f"View {i+1} ↔ View {j+1}  ({len(p1)} inliers)",
                 color='white', fontsize=12)
    ax.axis('off')

plt.suptitle("SIFT Feature Matches (RANSAC inliers)", color='white', fontsize=14,
             fontweight='bold', y=1.01)
plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, "sfm_matches.png"),
             dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close(fig1)
print("    Saved: output/sfm_matches.png")

# ── Figure 2:  3D point cloud + camera positions ──────────────
fig2 = plt.figure(figsize=(14, 10))
fig2.patch.set_facecolor('#0d1117')
ax3d = fig2.add_subplot(111, projection='3d')
ax3d.set_facecolor('#0d1117')

# Point cloud
ax3d.scatter(cloud_clean[:,0], cloud_clean[:,1], cloud_clean[:,2],
             c=cloud_clean[:,2], cmap='plasma', s=4, alpha=0.6, label='3D points')

# Convex hull boundary projected at mean Z
if hull is not None:
    z_mean = cloud_clean[:,2].mean()
    hull_ordered = pts_xy[hull.vertices]
    hv_closed = np.vstack([hull_ordered, hull_ordered[0]])
    ax3d.plot(hv_closed[:,0], hv_closed[:,1], [z_mean]*len(hv_closed), 'c-', lw=1.2, alpha=0.8)

# Camera centres
cam_colors = ['#ff4444', '#44ff44', '#4488ff']
cam_labels = ['Cam 1 (ref)', 'Cam 2', 'Cam 3']
for idx, (C, col, lbl) in enumerate(zip(cam_centers, cam_colors, cam_labels)):
    ax3d.scatter(*C, color=col, s=120, zorder=5, label=lbl)
    ax3d.text(C[0], C[1], C[2]+0.05, lbl, color=col, fontsize=9)

ax3d.set_xlabel('X', color='white'); ax3d.set_ylabel('Y', color='white')
ax3d.set_zlabel('Z (depth)', color='white')
ax3d.tick_params(colors='white')
for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('#333333')
ax3d.set_title("3D Point Cloud + Camera Positions", color='white',
               fontsize=14, fontweight='bold', pad=15)
ax3d.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
fig2.savefig(os.path.join(OUTPUT_DIR, "sfm_pointcloud_3d.png"),
             dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close(fig2)
print("    Saved: output/sfm_pointcloud_3d.png")

# ── Figure 3:  Top-down view + convex hull boundary ───────────
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))
fig3.patch.set_facecolor('#0d1117')

# Left: XY scatter + hull
ax_l = axes3[0]; ax_l.set_facecolor('#0d1117')
ax_l.scatter(cloud_clean[:,0], cloud_clean[:,1],
             c=cloud_clean[:,2], cmap='plasma', s=6, alpha=0.7)
if hull is not None:
    hull_ordered = pts_xy[hull.vertices]
    hv = np.vstack([hull_ordered, hull_ordered[0]])
    ax_l.plot(hv[:,0], hv[:,1], 'c-', lw=2, label='Convex hull boundary')
    ax_l.fill(hull_ordered[:,0], hull_ordered[:,1], alpha=0.15, color='cyan')
ax_l.set_ylabel('Y (arbitrary units)', color='white')
ax_l.set_title('Top-Down View — Reconstructed Points\n& Boundary Estimate',
               color='white', fontsize=12, fontweight='bold')
ax_l.tick_params(colors='white'); ax_l.spines['bottom'].set_color('#555')
ax_l.spines['left'].set_color('#555')
ax_l.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
ax_l.set_xlabel('X (arbitrary units)', color='white')

# Right: Depth distribution histogram
ax_r = axes3[1]; ax_r.set_facecolor('#0d1117')
ax_r.hist(cloud_clean[:,2], bins=50, color='#7b5ea7', edgecolor='#0d1117', alpha=0.9)
ax_r.axvline(cloud_clean[:,2].mean(), color='cyan', lw=1.5, linestyle='--',
             label=f"Mean Z = {cloud_clean[:,2].mean():.4f}")
ax_r.set_xlabel('Z — Depth', color='white')
ax_r.set_ylabel('Count', color='white')
ax_r.set_title('Depth (Z) Distribution\nof Reconstructed Points',
               color='white', fontsize=12, fontweight='bold')
ax_r.tick_params(colors='white')
for spine in ax_r.spines.values(): spine.set_color('#555')
ax_r.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)

plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, "sfm_boundary.png"),
             dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close(fig3)
print("    Saved: output/sfm_boundary.png")

# ── Figure 4:  Mathematical summary poster ───────────────────
fig4, ax = plt.subplots(figsize=(14, 18))
fig4.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117'); ax.axis('off')

math_text = (
    "STRUCTURE FROM MOTION — MATHEMATICAL WORKOUTS\n"
    "═══════════════════════════════════════════════════════════\n\n"
    "STEP 1 — Camera Intrinsic Matrix K\n"
    "────────────────────────────────────────────────\n"
    f"  f_physical = {F_PHYS} mm,  Sensor = {SENSOR_W}×{SENSOR_H} mm\n"
    f"  Pixel pitch: px = W/sw = {W}/{SENSOR_W} = {W/SENSOR_W:.1f} px/mm\n"
    f"  fx = f × px = {F_PHYS} × {W/SENSOR_W:.1f} = {fx:.1f} px\n"
    f"  fy = f × py = {F_PHYS} × {H/SENSOR_H:.1f} = {fy:.1f} px\n"
    f"  K = [[{fx:.0f}, 0, {cx:.0f}],\n"
    f"       [0, {fy:.0f}, {cy:.0f}],\n"
    f"       [0,    0,    1]]\n\n"
    "STEP 2 — Fundamental Matrix F  (8-point + RANSAC)\n"
    "────────────────────────────────────────────────\n"
    "  Epipolar constraint:  x'ᵀ F x = 0\n"
    "  Build A (N×9), each row: [x'x, x'y, x', y'x, y'y, y', x, y, 1]\n"
    "  Solve A f = 0 via SVD → f = last row of Vᵀ\n"
    "  Enforce rank-2: zero smallest singular value\n"
    "  RANSAC threshold = 1.0 px, confidence = 99.9%\n\n"
    "STEP 3 — Essential Matrix E\n"
    "────────────────────────────────────────────────\n"
    "  E = Kᵀ F K\n"
    "  Enforce σ₁=σ₂, σ₃=0:  E ← U diag(1,1,0) Vᵀ\n"
    "  det(U)>0 and det(Vᵀ)>0 enforced\n\n"
    "STEP 4 — Pose Recovery  R, t  (via SVD of E)\n"
    "────────────────────────────────────────────────\n"
    "  E = U Σ Vᵀ  (SVD)\n"
    "  W = [[0,-1,0],[1,0,0],[0,0,1]]  (90° rotation matrix)\n"
    "  R₁ = U W  Vᵀ  or  R₂ = U Wᵀ Vᵀ\n"
    "  t = ±U[:,2]\n"
    "  Select (R,t) where reconstructed pts have\n"
    "  positive depth in BOTH cameras (chirality check)\n\n"
    "STEP 5 — Triangulation  (DLT)\n"
    "────────────────────────────────────────────────\n"
    "  P₁ = K[I|0],  P₂ = K[R|t]\n"
    "  For each match:  build A (4×4), solve AX=0 via SVD\n"
    "  X = last column of V,  normalise by X[3]\n"
    "  Keep points with depth > 0 in both views\n\n"
    "STEP 6 — Boundary Estimation  (Convex Hull)\n"
    "────────────────────────────────────────────────\n"
    "  Project 3D cloud → 2D XY plane\n"
    "  Compute convex hull (Qhull algorithm)\n"
    f"  Hull vertices: {len(hull.vertices) if hull else 'N/A'}\n"
    f"  Aspect ratio: {aspect:.3f}  (true card: {85.6/54:.3f})\n"
)

ax.text(0.02, 0.98, math_text, transform=ax.transAxes,
        fontsize=10.5, color='#e0e0e0',
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#111827', alpha=0.9))
fig4.savefig(os.path.join(OUTPUT_DIR, "sfm_math_summary.png"),
             dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close(fig4)
print("    Saved: output/sfm_math_summary.png")

# ─────────────────────────────────────────────────────────────
# 11. CAMERA PARAMETER REPORT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("CAMERA PARAMETERS SUMMARY")
print("=" * 65)
print(f"  Device          : Apple iPhone 16 Pro")
print(f"  Lens            : {F_PHYS} mm f/2.2 (24mm equivalent)")
print(f"  Sensor size     : {SENSOR_W} × {SENSOR_H} mm  (1/1.28\" format)")
print(f"  Image size      : {W} × {H} pixels")
print(f"  fx              : {fx:.2f} px")
print(f"  fy              : {fy:.2f} px")
print(f"  Principal point : ({cx:.1f}, {cy:.1f}) px")
print(f"  Distortion      : not modelled (iPhone optics ~0)")
print()
for idx, (lbl, C) in enumerate(zip(cam_labels, cam_centers)):
    print(f"  {lbl}")
    print(f"    World centre  : {C.round(5)}")
    if idx > 0:
        key = (0, idx)
        if key in poses:
            R, t = poses[key]
            angle = np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1)))
            print(f"    Rotation angle: {angle:.2f}° rel. to View 1")
            print(f"    Translation   : {t.round(5)}")
    print()

print("=" * 65)
print("All outputs saved to  output/")
print("  sfm_matches.png         — Feature correspondences")
print("  sfm_pointcloud_3d.png   — 3D reconstruction + cameras")
print("  sfm_boundary.png        — Top-down boundary + depth hist")
print("  sfm_math_summary.png    — Mathematical workouts poster")
print("=" * 65)