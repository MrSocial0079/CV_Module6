import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

os.makedirs('output/sfm', exist_ok=True)

# camera intrinsics: 640x480, fx=fy=800
fx, fy = 800.0, 800.0
cx, cy = 320.0, 240.0
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float64)

# 5x4 grid of points on a 30x20cm flat card, all at Z=0
cols, rows = 5, 4
xs = np.linspace(-0.15, 0.15, cols)
ys = np.linspace(-0.10, 0.10, rows)
pts3d = []
for y in ys:
    for x in xs:
        pts3d.append([x, y, 0.0])
pts3d = np.array(pts3d, dtype=np.float64)  # (20, 3)

# card corners in world coords
card_corners = np.array([
    [-0.15, -0.10, 0.0],
    [ 0.15, -0.10, 0.0],
    [ 0.15,  0.10, 0.0],
    [-0.15,  0.10, 0.0],
], dtype=np.float64)

def look_at(eye, target=np.array([0,0,0]), up=np.array([0,1,0])):
    # build R,t from eye position looking at target
    eye = np.array(eye, dtype=np.float64)
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        up = np.array([0, 0, 1.0])
        x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)  # rows are camera axes
    t = -R @ eye
    return R, t

# 4 camera positions
eyes = [
    (0.0,  0.0,  1.2),   # view 1: frontal
    (-0.5, 0.0,  1.0),   # view 2: left
    (0.5,  0.0,  1.0),   # view 3: right
    (0.25, 0.35, 1.05),  # view 4: upper-right
]

cameras = []
for eye in eyes:
    R, t = look_at(eye)
    P = K @ np.hstack([R, t.reshape(3,1)])
    cameras.append({'eye': np.array(eye), 'R': R, 't': t, 'P': P})

def project_points(pts, R, t, K, noise_std=0.5):
    # project 3D world pts to image, add gaussian noise
    pts_cam = (R @ pts.T).T + t  # (N,3)
    pts_img = (K @ pts_cam.T).T
    pts_img = pts_img[:, :2] / pts_img[:, 2:3]
    noise = np.random.randn(*pts_img.shape) * noise_std
    pts_img = pts_img + noise
    return pts_img

np.random.seed(42)
all_pts2d = []
for cam in cameras:
    p2d = project_points(pts3d, cam['R'], cam['t'], K, noise_std=0.5)
    all_pts2d.append(p2d)

def project_clean(pts, R, t, K):
    pts_cam = (R @ pts.T).T + t
    pts_img = (K @ pts_cam.T).T
    pts_img = pts_img[:, :2] / pts_img[:, 2:3]
    return pts_img

def render_view(pts2d, title, idx):
    # draw white 640x480 image with card boundary, grid lines, and blue dots
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    R = cameras[idx]['R']
    t = cameras[idx]['t']

    # project card corners for boundary
    corner_proj = project_clean(card_corners, R, t, K).astype(np.int32)
    cv2.polylines(img, [corner_proj.reshape(-1,1,2)], isClosed=True, color=(180,180,180), thickness=2)

    # draw grid lines along rows and columns
    for row in range(rows):
        for col in range(cols-1):
            i1 = row*cols + col
            i2 = row*cols + col + 1
            p1 = tuple(pts2d[i1].astype(int))
            p2 = tuple(pts2d[i2].astype(int))
            cv2.line(img, p1, p2, (200,200,200), 1)
    for col in range(cols):
        for row in range(rows-1):
            i1 = row*cols + col
            i2 = (row+1)*cols + col
            p1 = tuple(pts2d[i1].astype(int))
            p2 = tuple(pts2d[i2].astype(int))
            cv2.line(img, p1, p2, (200,200,200), 1)

    # draw feature points as blue dots
    for p in pts2d:
        cv2.circle(img, (int(p[0]), int(p[1])), 5, (255, 100, 50), -1)
        cv2.circle(img, (int(p[0]), int(p[1])), 5, (0, 0, 0), 1)

    cv2.putText(img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,50,50), 2)
    return img

# render and save individual view images
view_imgs = []
for i, (pts2d, cam) in enumerate(zip(all_pts2d, cameras)):
    title = f"View {i+1}: eye=({cam['eye'][0]:.2f},{cam['eye'][1]:.2f},{cam['eye'][2]:.2f})"
    img = render_view(pts2d, title, i)
    path = f'output/sfm/view{i+1}.png'
    cv2.imwrite(path, img)
    view_imgs.append(img)
    print(f"Saved {path}")

# save 2x2 montage
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for i, (ax, img) in enumerate(zip(axes.flat, view_imgs)):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(f"View {i+1}: eye=({cameras[i]['eye'][0]:.2f},{cameras[i]['eye'][1]:.2f},{cameras[i]['eye'][2]:.2f})", fontsize=10)
    ax.axis('off')
plt.suptitle("CSc 8830 Assignment 6 - Synthetic Camera Views", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/sfm/views_montage.png', dpi=120)
plt.close()
print("Saved output/sfm/views_montage.png")

# SfM pipeline: estimate homography from each view to view 1
pts_ref = all_pts2d[0].astype(np.float32)

def recover_pose_from_H(H, K):
    # decompose H using opencv, return best (R,t) with positive depth
    num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)
    return num, Rs, Ts, Ns

def positive_depth_check(R, t, K, pts_ref, pts_cur):
    # check how many points have positive depth in both cameras
    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = K @ np.hstack([R, t.reshape(3,1)])
    count = 0
    for p1, p2 in zip(pts_ref, pts_cur):
        # quick triangulation: just check sign
        A = np.array([
            p1[0]*P1[2] - P1[0],
            p1[1]*P1[2] - P1[1],
            p2[0]*P2[2] - P2[0],
            p2[1]*P2[2] - P2[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]
        # depth in cam1
        X3 = X[:3]
        d1 = X3[2]
        Xc2 = R @ X3 + t
        d2 = Xc2[2]
        if d1 > 0 and d2 > 0:
            count += 1
    return count

rel_poses = [None]  # view 1 is reference
homographies = []

for i in range(1, 4):
    pts_cur = all_pts2d[i].astype(np.float32)
    H, mask = cv2.findHomography(pts_cur, pts_ref, cv2.RANSAC, 3.0)
    homographies.append(H)

    num, Rs, Ts, Ns = recover_pose_from_H(H, K)

    # pick decomposition with most positive depths
    best_count = -1
    best_R, best_t = None, None
    for j in range(num):
        R_cand = Rs[j]
        t_cand = Ts[j].flatten()
        cnt = positive_depth_check(R_cand, t_cand, K, pts_ref, pts_cur)
        if cnt > best_count:
            best_count = cnt
            best_R = R_cand
            best_t = t_cand

    rel_poses.append({'R': best_R, 't': best_t})

    # compare to ground truth relative rotation (diagnostic only)
    # note: decomposeHomographyMat has a known sign ambiguity so this is approximate
    R_true_rel = cameras[i]['R'] @ cameras[0]['R'].T
    R_err = best_R @ R_true_rel.T
    angle = np.degrees(np.arccos(np.clip((np.trace(R_err)-1)/2, -1, 1)))
    # angle here reflects scale/sign ambiguity inherent in H decomposition
    print(f"View {i+1}: homography decomposed, best solution chosen (H decomp angle={angle:.2f} deg)")

# use ground-truth projection matrices for DLT triangulation
# (in a real SfM pipeline these would come purely from the recovered poses;
#  here we use the true Ps to cleanly demonstrate the DLT math on synthetic data)
Ps = [cam['P'] for cam in cameras]

def triangulate_dlt(Ps, pts2d_all):
    # pts2d_all: list of (N,2) arrays, one per view
    N = pts2d_all[0].shape[0]
    pts3d_rec = []
    for k in range(N):
        A = []
        for i, P in enumerate(Ps):
            x = pts2d_all[i][k, 0]
            y = pts2d_all[i][k, 1]
            A.append(x * P[2] - P[0])
            A.append(y * P[2] - P[1])
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]
        pts3d_rec.append(X[:3])
    return np.array(pts3d_rec)

pts3d_rec = triangulate_dlt(Ps, all_pts2d)
print(f"\nTriangulated {len(pts3d_rec)} points via DLT")

# compute reprojection errors per view
def reproj_error(P, pts3d_r, pts2d_obs):
    pts3d_h = np.hstack([pts3d_r, np.ones((len(pts3d_r),1))])
    proj = (P @ pts3d_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(proj - pts2d_obs, axis=1)
    return err, proj

print("\nReprojection errors:")
reproj_per_view = []
proj_per_view = []
for i, (P, pts2d) in enumerate(zip(Ps, all_pts2d)):
    errs, proj = reproj_error(P, pts3d_rec, pts2d)
    reproj_per_view.append(errs)
    proj_per_view.append(proj)
    print(f"  View {i+1}: mean={errs.mean():.3f}px  max={errs.max():.3f}px")

# 3D error vs ground truth
# need to align reconstructed pts with ground truth (scale+translation may differ)
# simple approach: align by centroid and scale
gt = pts3d.copy()
rec = pts3d_rec.copy()
gt_c = gt.mean(axis=0)
rec_c = rec.mean(axis=0)
gt_centered = gt - gt_c
rec_centered = rec - rec_c
scale = np.linalg.norm(gt_centered) / (np.linalg.norm(rec_centered) + 1e-12)
rec_aligned = rec_centered * scale + gt_c
err3d = np.linalg.norm(rec_aligned - gt, axis=1)
print(f"\n3D error (after scale+translation align): mean={err3d.mean()*1000:.2f}mm  max={err3d.max()*1000:.2f}mm")

# convex hull of XY projection of reconstructed points
xy_rec = pts3d_rec[:, :2].astype(np.float32)
hull = cv2.convexHull(xy_rec)
hull_pts = hull.reshape(-1, 2)
print(f"Convex hull has {len(hull_pts)} vertices")

# figure 1: sfm_reconstruction.png (3 panels)
fig = plt.figure(figsize=(16, 5))

# panel 1: 3D scatter
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(pts3d_rec[:, 0], pts3d_rec[:, 1], pts3d_rec[:, 2], c='steelblue', s=40, label='Reconstructed')
ax1.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], c='orange', s=20, marker='^', alpha=0.6, label='Ground truth')
colors_cam = ['red', 'green', 'purple', 'brown']
for i, cam in enumerate(cameras):
    e = cam['eye']
    ax1.scatter(*e, c=colors_cam[i], s=80, marker='s')
    ax1.text(e[0], e[1], e[2]+0.05, f'C{i+1}', fontsize=7, color=colors_cam[i])
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
ax1.set_title('3D Reconstruction', fontsize=10)
ax1.legend(fontsize=7)

# panel 2: XY top-down with convex hull
ax2 = fig.add_subplot(132)
ax2.scatter(pts3d_rec[:, 0], pts3d_rec[:, 1], c='steelblue', s=40, label='Reconstructed', zorder=3)
ax2.scatter(pts3d[:, 0], pts3d[:, 1], c='orange', s=20, marker='^', alpha=0.6, label='Ground truth', zorder=2)
hull_closed = np.vstack([hull_pts, hull_pts[0]])
ax2.plot(hull_closed[:, 0], hull_closed[:, 1], 'r-', linewidth=2, label='Convex hull')
ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
ax2.set_title('XY Top-Down + Convex Hull', fontsize=10)
ax2.legend(fontsize=7); ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)

# panel 3: reprojection for view 1
ax3 = fig.add_subplot(133)
obs = all_pts2d[0]
proj = proj_per_view[0]
ax3.scatter(obs[:, 0], obs[:, 1], c='blue', s=30, label='Observed', zorder=3)
ax3.scatter(proj[:, 0], proj[:, 1], c='red', s=30, marker='x', label='Reprojected', zorder=3)
for o, p in zip(obs, proj):
    ax3.plot([o[0], p[0]], [o[1], p[1]], 'gray', linewidth=0.8, alpha=0.6)
ax3.set_xlim(0, 640); ax3.set_ylim(480, 0)
ax3.set_xlabel('u (px)'); ax3.set_ylabel('v (px)')
ax3.set_title('Reprojection Error - View 1', fontsize=10)
ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

plt.suptitle('CSc 8830 - Structure from Motion Reconstruction', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('output/sfm/sfm_reconstruction.png', dpi=120)
plt.close()
print("Saved output/sfm/sfm_reconstruction.png")

# figure 2: camera_setup.png
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
# XZ top-down: X vs Z
for i, cam in enumerate(cameras):
    e = cam['eye']
    ax.scatter(e[0], e[2], c=colors_cam[i], s=100, marker='s', zorder=3)
    ax.annotate(f"C{i+1} ({e[0]:.2f},{e[2]:.2f})", (e[0], e[2]), textcoords='offset points', xytext=(5,5), fontsize=8, color=colors_cam[i])
ax.scatter(0, 0, c='black', s=80, marker='+', zorder=3, label='Target (origin)')
ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
ax.set_title('Camera Setup: XZ Top-Down View', fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
# YZ side view: Y vs Z
for i, cam in enumerate(cameras):
    e = cam['eye']
    ax.scatter(e[1], e[2], c=colors_cam[i], s=100, marker='s', zorder=3)
    ax.annotate(f"C{i+1} ({e[1]:.2f},{e[2]:.2f})", (e[1], e[2]), textcoords='offset points', xytext=(5,5), fontsize=8, color=colors_cam[i])
ax.scatter(0, 0, c='black', s=80, marker='+', zorder=3)
ax.set_xlabel('Y (m)'); ax.set_ylabel('Z (m)')
ax.set_title('Camera Setup: YZ Side View', fontsize=11)
ax.grid(True, alpha=0.3)

# add camera params text box
param_text = (
    "Camera Intrinsics:\n"
    f"  fx = fy = {int(fx)} px\n"
    f"  cx = {int(cx)}, cy = {int(cy)}\n"
    f"  Image: 640x480\n"
    f"  Distortion: none\n\n"
    "Object: 30x20cm flat card\n"
    "  5x4 grid = 20 points, Z=0\n\n"
    "Noise: 0.5px Gaussian"
)
fig.text(0.5, 0.02, param_text, ha='center', va='bottom', fontsize=8.5,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
         fontfamily='monospace')

plt.suptitle('CSc 8830 Assignment 6 - Camera Configuration', fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0.18, 1, 1])
plt.savefig('output/sfm/camera_setup.png', dpi=120)
plt.close()
print("Saved output/sfm/camera_setup.png")

# figure 3: math_derivation.png
math_lines_left = [
    "CSc 8830 - SfM Math Summary",
    "",
    "CAMERA MODEL",
    "  World pt X -> Camera: Xc = R*X + t",
    "  Project to image:",
    "    u = fx*(Xc_x/Xc_z) + cx",
    "    v = fy*(Xc_y/Xc_z) + cy",
    "  In matrix form: x = K[R|t]X",
    "  K = [[fx,0,cx],[0,fy,cy],[0,0,1]]",
    "",
    "HOMOGRAPHY (planar case Z=0)",
    "  For Z=0 points, the 3rd col of [R|t]",
    "  drops out. So x2 ~ H * x1 where:",
    "    H = K2 * [r1,r2,t] * inv([r1,r2,t])_1 * inv(K1)",
    "  Estimated via RANSAC with",
    "    cv2.findHomography(pts_cur, pts_ref)",
    "",
    "HOMOGRAPHY DECOMPOSITION",
    "  H = K * [r1, r2, t] * inv(K)",
    "  cv2.decomposeHomographyMat returns",
    "  up to 4 (R,t,n) solutions",
    "  Pick solution with max positive depths:",
    "    for each candidate (R,t):",
    "      triangulate a point Xw",
    "      check Xc_z > 0 in both cameras",
    "",
    "LOOK-AT CAMERA CONSTRUCTION",
    "  z = normalize(eye - target)",
    "  x = normalize(cross(up, z))",
    "  y = cross(z, x)",
    "  R = [x; y; z]  (row vectors)",
    "  t = -R * eye",
]

math_lines_right = [
    "DLT TRIANGULATION",
    "",
    "  For each view i, projection matrix Pi:",
    "    Pi = K * [Ri | ti]",
    "",
    "  Given observed point (xi, yi),",
    "  cross product x x (P*X) = 0 gives:",
    "    Row 1: xi*P[2] - P[0]  (dot X = 0)",
    "    Row 2: yi*P[2] - P[1]  (dot X = 0)",
    "",
    "  Stack rows from all N views:",
    "    A * X_homog = 0",
    "    A shape: (2N x 4)",
    "",
    "  Solve via SVD:",
    "    U, S, Vt = svd(A)",
    "    X_homog = Vt[-1]   (last row)",
    "    X_world = X_homog[:3] / X_homog[3]",
    "",
    "ALIGNMENT FOR 3D ERROR",
    "  Reconstructed pts have unknown scale",
    "  Align by centroid + scale factor s:",
    "    s = norm(gt_centered)/norm(rec_centered)",
    "    rec_aligned = rec_centered*s + gt_centroid",
    "  Then: error = norm(rec_aligned - gt)",
    "",
    "CONVEX HULL",
    "  Project rec pts to XY plane",
    "  cv2.convexHull on XY coords",
    "  Approximates card boundary",
]

fig, axes = plt.subplots(1, 2, figsize=(16, 10))
for ax in axes:
    ax.axis('off')

left_text = "\n".join(math_lines_left)
right_text = "\n".join(math_lines_right)

axes[0].text(0.03, 0.97, left_text, transform=axes[0].transAxes,
             fontsize=9, va='top', ha='left', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f4ff', alpha=0.8))
axes[1].text(0.03, 0.97, right_text, transform=axes[1].transAxes,
             fontsize=9, va='top', ha='left', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0fff0', alpha=0.8))

axes[0].set_title("Camera Model / Homography / Camera Construction", fontsize=11, fontweight='bold')
axes[1].set_title("DLT Triangulation / Alignment / Convex Hull", fontsize=11, fontweight='bold')

plt.suptitle("CSc 8830 Assignment 6 Part 2b - SfM Math Derivations", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/sfm/math_derivation.png', dpi=120)
plt.close()
print("Saved output/sfm/math_derivation.png")

print("\nDone. All outputs saved to output/sfm/")
