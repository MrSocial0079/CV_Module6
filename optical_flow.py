# CSc 8830 - Computer Vision, Assignment 6 - Part 1
# Optical flow: dense (Farneback) and sparse (Lucas-Kanade) on two videos.
# Usage: python optical_flow.py
# Outputs saved to output/optical_flow/

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

VIDEO_PATHS = {
    "video1": "videos/video1.mp4",
    "video2": "videos/video2.mp4",
}
OUTPUT_DIR = "output/optical_flow"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Farneback parameters
FARN_PARAMS = dict(
    pyr_scale  = 0.5,
    levels     = 3,
    winsize    = 15,
    iterations = 3,
    poly_n     = 5,
    poly_sigma = 1.2,
    flags      = 0,
)

# Lucas-Kanade sparse tracker parameters
LK_PARAMS = dict(
    winSize  = (21, 21),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# Shi-Tomasi corner detector — seed points for LK
FEATURE_PARAMS = dict(
    maxCorners   = 200,
    qualityLevel = 0.01,
    minDistance  = 10,
    blockSize    = 7,
)

SAMPLE_START_SEC = 2   # skip first N seconds
MAX_FRAMES = None      # set a number to limit processing
PROCESS_WIDTH = 720    # downscale wide/4K frames before computing flow


def resize_frame(frame):
    if PROCESS_WIDTH is None:
        return frame
    h, w = frame.shape[:2]
    if w <= PROCESS_WIDTH:
        return frame
    scale = PROCESS_WIDTH / w
    return cv2.resize(frame, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


# Encode dense flow as HSV: hue = direction, value = magnitude
def flow_to_hsv(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# Draw a grid of arrows representing the flow field
def draw_flow_arrows(frame, flow, step=16, scale=3.0):
    vis = frame.copy()
    h, w = frame.shape[:2]
    y_coords, x_coords = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx = flow[y_coords, x_coords, 0]
    fy = flow[y_coords, x_coords, 1]
    for x, y, u, v in zip(x_coords, y_coords, fx, fy):
        cv2.arrowedLine(vis, (x, y),
                        (int(x + u * scale), int(y + v * scale)),
                        (0, 255, 0), 1, tipLength=0.3)
    return vis


def compute_dense_flow(video_name, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(SAMPLE_START_SEC * fps))

    out_path = os.path.join(OUTPUT_DIR, f"{video_name}_dense_flow.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"avc1")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return
    prev_frame = resize_frame(prev_frame)
    prev_gray  = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    height, width = prev_frame.shape[:2]
    writer.release()
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_count    = 0
    analysis_saved = False
    magnitudes     = []

    print(f"\n[Dense Flow] {video_name}  ({width}x{height})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if MAX_FRAMES and frame_count >= MAX_FRAMES:
            break

        frame     = resize_frame(frame)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                            **FARN_PARAMS)

        hsv_vis   = flow_to_hsv(flow)
        arrow_vis = draw_flow_arrows(frame, flow, step=20, scale=4.0)

        # Side-by-side composite: original left, flow right
        lo = frame.copy()
        lf = hsv_vis.copy()
        cv2.putText(lo, "Original",         (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(lf, "Dense Flow (HSV)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        composite = cv2.resize(np.hstack([lo, lf]), (width, height))
        writer.write(composite)

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(mag.mean())

        if not analysis_saved and mag.mean() > 0.5:
            _save_dense_analysis(video_name, frame, flow, hsv_vis, arrow_vis)
            analysis_saved = True

        prev_gray = curr_gray
        frame_count += 1

    cap.release()
    writer.release()
    print(f"  Frames: {frame_count}  Mean mag: {np.mean(magnitudes):.3f} px/frame")
    print(f"  Saved : {out_path}")


def _save_dense_analysis(video_name, frame, flow, hsv_vis, arrow_vis):
    # 6-panel figure: original, HSV flow, arrows, magnitude, u-component, v-component
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    u = flow[..., 0]
    v = flow[..., 1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"{video_name} — Dense Optical Flow Analysis (Farneback)",
                 fontsize=14, fontweight="bold")

    axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("(a) Original Frame"); axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(hsv_vis, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("(b) HSV Flow  [Hue=Direction, Brightness=Speed]"); axes[0, 1].axis("off")

    axes[0, 2].imshow(cv2.cvtColor(arrow_vis, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("(c) Arrow Overlay"); axes[0, 2].axis("off")

    im = axes[1, 0].imshow(mag, cmap="hot", interpolation="nearest")
    axes[1, 0].set_title("(d) Flow Magnitude"); axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im_u = axes[1, 1].imshow(u, cmap="bwr", interpolation="nearest",
                              vmin=-mag.max(), vmax=mag.max())
    axes[1, 1].set_title("(e) Horizontal u  [red=right, blue=left]"); axes[1, 1].axis("off")
    plt.colorbar(im_u, ax=axes[1, 1], fraction=0.046)

    im_v = axes[1, 2].imshow(v, cmap="bwr", interpolation="nearest",
                              vmin=-mag.max(), vmax=mag.max())
    axes[1, 2].set_title("(f) Vertical v  [red=down, blue=up]"); axes[1, 2].axis("off")
    plt.colorbar(im_v, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{video_name}_dense_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Analysis: {out}")


def compute_sparse_flow(video_name, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(SAMPLE_START_SEC * fps))

    out_path = os.path.join(OUTPUT_DIR, f"{video_name}_sparse_flow.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"avc1")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return
    prev_frame = resize_frame(prev_frame)
    prev_gray  = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    height, width = prev_frame.shape[:2]
    writer.release()
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)

    np.random.seed(42)
    colors    = np.random.randint(50, 255, (500, 3), dtype=np.uint8)
    N_TRAIL   = 20
    trails    = {}
    track_ids = {}
    next_id   = 0

    if p0 is not None:
        for i, pt in enumerate(p0):
            track_ids[i] = next_id
            trails[next_id] = [tuple(pt[0].astype(int))]
            next_id += 1

    frame_count    = 0
    REDETECT_EVERY = 30

    print(f"\n[Sparse Flow] {video_name}  ({width}x{height})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if MAX_FRAMES and frame_count >= MAX_FRAMES:
            break

        frame     = resize_frame(frame)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis       = frame.copy()

        if p0 is not None and len(p0) > 0:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray,
                                                  p0, None, **LK_PARAMS)
            # back-track to filter unreliable tracks
            if p1 is not None:
                p0r, _, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray,
                                                      p1, None, **LK_PARAMS)
                d    = abs(p0 - p0r).reshape(-1, 2).max(axis=1)
                good = (st.flatten() == 1) & (d < 1.0)
            else:
                good = np.zeros(len(p0), dtype=bool)

            new_p0, new_ids, new_idx = [], {}, 0

            for i, (is_good, new_pt) in enumerate(zip(good, p1 if p1 is not None else [])):
                if not is_good:
                    continue
                tid  = track_ids.get(i, next_id)
                x, y = int(new_pt[0, 0]), int(new_pt[0, 1])

                if tid not in trails:
                    trails[tid] = []
                trails[tid].append((x, y))
                if len(trails[tid]) > N_TRAIL:
                    trails[tid].pop(0)

                new_p0.append(new_pt)
                new_ids[new_idx] = tid
                new_idx += 1

                col = [int(c) for c in colors[tid % 500]]
                cv2.circle(vis, (x, y), 4, col, -1)

            for tid, pts in trails.items():
                col = [int(c) for c in colors[tid % 500]]
                for j in range(1, len(pts)):
                    alpha = j / len(pts)
                    cv2.line(vis, pts[j-1], pts[j],
                             [int(v * alpha) for v in col], 2)

            p0        = np.array(new_p0, dtype=np.float32) if new_p0 else None
            track_ids = new_ids

        n_tracked = len(p0) if p0 is not None else 0

        # periodically re-detect to replace lost features
        if frame_count % REDETECT_EVERY == 0 or n_tracked < 20:
            new_pts = cv2.goodFeaturesToTrack(curr_gray, mask=None, **FEATURE_PARAMS)
            if new_pts is not None:
                if p0 is not None and len(p0) > 0:
                    merged = []
                    for pt in new_pts:
                        xn, yn = pt[0]
                        dists = np.sqrt(((p0[:, 0, 0] - xn)**2 +
                                         (p0[:, 0, 1] - yn)**2))
                        if dists.min() > 10:
                            merged.append(pt)
                    if merged:
                        new_arr = np.array(merged, dtype=np.float32)
                        p0 = np.concatenate([p0, new_arr], axis=0)
                        for k in range(len(new_arr)):
                            track_ids[n_tracked + k] = next_id
                            trails[next_id] = [tuple(new_arr[k, 0].astype(int))]
                            next_id += 1
                else:
                    p0 = new_pts
                    for k, pt in enumerate(new_pts):
                        track_ids[k] = next_id
                        trails[next_id] = [tuple(pt[0].astype(int))]
                        next_id += 1

        cv2.putText(vis, f"Sparse LK  tracks:{n_tracked}  f:{frame_count}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        writer.write(vis)
        prev_gray = curr_gray
        frame_count += 1

    cap.release()
    writer.release()
    print(f"  Frames: {frame_count}")
    print(f"  Saved : {out_path}")


def save_inference_summary(video_name, video_path):
    # Pick two frame pairs (early and mid-video) and show flow side by side
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    sf    = int(SAMPLE_START_SEC * fps)
    pairs = [(sf, sf + 1), (total // 2, total // 2 + 1)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"{video_name} — What Optical Flow Tells Us\n"
        "(Hue=Direction, Brightness=Magnitude; black=no motion)",
        fontsize=13, fontweight="bold"
    )

    for row, (f1, f2) in enumerate(pairs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f1)
        ret1, fr1 = cap.read()
        ret2, fr2 = cap.read()
        if not ret1 or not ret2:
            continue

        fr1  = resize_frame(fr1)
        fr2  = resize_frame(fr2)
        g1   = cv2.cvtColor(fr1, cv2.COLOR_BGR2GRAY)
        g2   = cv2.cvtColor(fr2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, **FARN_PARAMS)

        hsv_vis   = flow_to_hsv(flow)
        arrow_vis = draw_flow_arrows(fr1, flow, step=18, scale=5.0)
        mag, _    = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        axes[row, 0].imshow(cv2.cvtColor(fr1, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title(f"Original (t={f1/fps:.1f}s)"); axes[row, 0].axis("off")

        axes[row, 1].imshow(cv2.cvtColor(hsv_vis, cv2.COLOR_BGR2RGB))
        axes[row, 1].set_title("Dense Flow (HSV)"); axes[row, 1].axis("off")

        axes[row, 2].imshow(cv2.cvtColor(arrow_vis, cv2.COLOR_BGR2RGB))
        axes[row, 2].set_title(f"Arrow Field  (mean={mag.mean():.2f}px)")
        axes[row, 2].axis("off")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{video_name}_inference_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    cap.release()
    print(f"  Summary : {out}")


def main():
    for name, path in VIDEO_PATHS.items():
        if not os.path.isfile(path):
            print(f"[SKIP] {path} not found")
            continue
        print(f"\n{'='*55}\n  {name}  ({path})\n{'='*55}")
        compute_dense_flow(name, path)
        compute_sparse_flow(name, path)
        save_inference_summary(name, path)
    print("\n[Done] Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
