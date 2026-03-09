# CSc 8830: Computer Vision — Assignment 6
## Optical Flow & Motion Tracking
**Author:** Yasaswi Kompella

---

### Repository Structure
```
CV module 6/
├── README.md
├── optical_flow.py        # Part 1: dense + sparse optical flow
├── motion_tracking.py     # Part 2: LK tracking from scratch + bilinear interp
├── videos/
│   ├── video1.mp4         # Swimmers (translational + turbulent motion)
│   └── video2.mp4         # Aerial traffic (multi-directional motion)
└── output/
    ├── optical_flow/      # flow videos and analysis figures
    └── tracking/          # tracking validation figures and report
```

---

### Dependencies
```
pip install opencv-contrib-python numpy matplotlib scipy
```

---

### How to Run

**Part 1 — Optical Flow**
```bash
python optical_flow.py
```
Outputs:
- `videoN_dense_flow.mp4` — side-by-side original + HSV-encoded Farneback flow
- `videoN_sparse_flow.mp4` — Lucas-Kanade feature trails
- `videoN_dense_analysis.png` — 6-panel: original, HSV flow, arrows, magnitude, u, v
- `videoN_inference_summary.png` — early vs mid-video flow comparison

**Part 2 — Motion Tracking**
```bash
python motion_tracking.py
```
Outputs:
- `videoN_tracking_validation.png` — 4-panel: seeds, our LK, OpenCV LK, overlay
- `bilinear_interp_demo.png` — nearest-neighbour vs bilinear + formula diagram
- `tracking_report.txt` — numerical point-by-point comparison table

---

### Part 1: Optical Flow

**Dense Flow (Farneback)**
Computes a 2D flow vector (u, v) at every pixel using polynomial expansion across a Gaussian pyramid. Visualized as an HSV image:
- Hue → direction of motion
- Brightness → speed (magnitude)

**Sparse Flow (Lucas-Kanade)**
Detects Shi-Tomasi corner features and tracks them frame-to-frame using pyramidal LK. Points are drawn with colored trails showing their path over the last 20 frames.

**What optical flow tells us:**
- Uniform flow across the frame → camera translation
- Radially expanding flow from center → camera zoom in / forward motion
- Circular patterns → rotation
- Flow discontinuities at object edges → foreground/background boundaries
- Regions with zero flow → static background
- Different magnitudes in different regions → objects moving at different speeds

---

### Part 2: Motion Tracking Derivation

**Brightness Constancy Assumption:**
```
I(x, y, t) = I(x+u, y+v, t+1)
```

**Taylor expansion (linearization):**
```
I2(x+u, y+v) ≈ I2(x,y) + Ix*u + Iy*v
=> Ix*u + Iy*v + It = 0    (Optical Flow Constraint Equation)
```
One equation, two unknowns — the aperture problem. Cannot solve from a single pixel.

**Patch assumption — overdetermined system:**
```
For N pixels in window W:   A * d = -b
A = [[Ix1,Iy1],[Ix2,Iy2],...],   b = [It1, It2, ...]^T,   d = [u,v]^T
```

**Least-squares solution (normal equations):**
```
d = -(A^T A)^-1 A^T b
```
where `M = A^T A = [[SIx2, SIxIy],[SIxIy, SIy2]]` is the structure tensor.
M must be invertible (both eigenvalues large) — same condition as corner detection.

**Iterative refinement:** warp I2 by current estimate, compute residual, update until ||delta|| < epsilon.

**Pyramidal extension:** build Gaussian pyramid; track coarse-to-fine so large motions become small displacements at coarser levels.

**Bilinear Interpolation:**
When a tracked point lands at a sub-pixel location (x+dx, y+dy):
```
I = (1-dx)(1-dy)*I(x,y) + dx(1-dy)*I(x+1,y)
  + (1-dx)*dy*I(x,y+1)  + dx*dy*I(x+1,y+1)
```

---

### References
1. Lucas, B.D. & Kanade, T. (1981). An Iterative Image Registration Technique with an Application to Stereo Vision. *IJCAI-81*, pp. 674–679.
2. Farneback, G. (2003). Two-Frame Motion Estimation Based on Polynomial Expansion. *SCIA 2003*, LNCS 2749, pp. 363–370.
3. Shi, J. & Tomasi, C. (1994). Good Features to Track. *CVPR 1994*.
4. Baker, S. & Matthews, I. (2004). Lucas-Kanade 20 Years On. *IJCV 56(3)*, pp. 221–255.
5. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd ed. Springer. Chapter 8.
6. OpenCV Optical Flow documentation: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
