# Pipeline

Every script in the repository, what it does, and where it sits in the measurement
chain. Filenames are as originally written; this document is the index.

---

## Phase 0 · Projector targets

The DLP LightCrafter runs at 640×360, so every generated pattern is authored at
exactly that resolution and letterboxed rather than scaled — resampling a calibration
target defeats the point of having one.

| File | Role |
|---|---|
| [`patterns/checkerboardgenerator.py`](../patterns/checkerboardgenerator.py) | Pixel-perfect checkerboard, centred on a 640×360 canvas. Prompts for inner-corner counts and square size, and shrinks the squares automatically if the requested grid would overflow the frame. Saves PNG + BMP with a random token in the name so successive attempts never collide. |
| [`patterns/circlegeneratorforangle.py`](../patterns/circlegeneratorforangle.py) | Asymmetric circle grid — the target for projector pose. Odd rows are offset by half a column pitch, which is what removes the 180° pose ambiguity. Emits 24-bit RGB (some projectors mishandle 8-bit indexed input) and verifies the channel count after saving. |
| [`patterns/circulardistortionreverserengineering.py`](../patterns/circulardistortionreverserengineering.py) | **Keystone simulator.** Builds a 3D rotation from chosen pitch/yaw/roll, projects the pattern's four corners through a virtual pinhole at a chosen distance, and warps the image by the resulting homography. Arrow-key HUD; `s` saves the clean warp with the angles in the filename. |

The simulator is the validation instrument. It produces images whose true pose is
known exactly, so any error the pose solver reports on them is algorithmic, not
experimental. Without it, a wrong angle is ambiguous between a bad solver, a bad
detection and a badly aligned bench.

---

## Phase 1 · Camera intrinsics

**[`calibration/Gen1checkerboarddistortion.py`](../calibration/Gen1checkerboarddistortion.py)**

An interactive Brown–Conrady tuner. Up/down selects a parameter, left/right adjusts
it, and the undistorted image re-renders every frame:

| Parameter | Meaning |
|---|---|
| `fx`, `fy` | Focal lengths in pixels |
| `cx`, `cy` | Principal point |
| `k1`, `k2` | Radial distortion — barrel/pincushion |
| `p1`, `p2` | Tangential distortion — sensor/lens decentring |
| `alpha` | Free scaling passed to `getOptimalNewCameraMatrix`; controls how much invalid border is cropped |

The radial and tangential terms are stored with a +1000 offset internally so the HUD
can hold them as integers, and are mapped back to physical values on display
(`k = (v − 1000)/1000`, `p = (v − 1000)/5000`).

Pressing `m` enters **measurement mode**: click two points a known number of
checkerboard squares apart, enter the square size in millimetres, and the tool
computes `mm_per_pixel`. That single scalar is what makes every later height a
physical measurement instead of a pixel count.

`s` writes `my_lens_metrics_<token>.json` to `~/Documents/`:

```json
{
  "camera_matrix":            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "distortion_coefficients":  [[k1, k2, p1, p2, 0.0]],
  "optimal_matrix":           [[...]],
  "roi":                      [x, y, w, h],
  "mm_per_pixel":             1.0
}
```

A worked example is [`calibration/example_lens_metrics.json`](../calibration/example_lens_metrics.json)
(and `assets/figures/jsonexample.png` shows it in a JSON viewer). Every downstream
script reads this file through the same `load_camera_metrics` helper.

**Why tune by hand rather than run `calibrateCamera`?** The target is *projected* onto
a dark, partly specular surface rather than printed on card. Automatic corner
detection is unreliable there, and a silently bad automatic fit is worse than a
deliberately supervised one — the tuner keeps the undistorted result in front of you
while you set the coefficients.

---

## Phase 2 · Projector extrinsics

**[`calibration/Gen1angulardetermination.py`](../calibration/Gen1angulardetermination.py)**

Recovers the projector's orientation relative to the fluid plane. This angle is the
single most important number in the reconstruction: height is inferred from fringe
displacement, and displacement converts to height through this angle's tangent.

1. **Undistort** using the Phase 1 metrics.
2. **Detect** the asymmetric circle grid with `cv2.findCirclesGrid(..., CALIB_CB_ASYMMETRIC_GRID)`,
   backed by a `SimpleBlobDetector` configured for bright blobs with circularity,
   convexity and inertia filters *disabled* — keystone stretches circles into ellipses,
   and shape filters would reject exactly the blobs that carry the pose information.
3. **Retry across channels** — green, high-contrast grayscale, plain grayscale, red,
   blue — until one locks. Detection on a glowing pattern reflected off dark liquid is
   the fragile step; the script reports which channel succeeded and, on total failure,
   renders what the blob detector actually saw.
4. **Solve** with `cv2.solvePnP` against the ideal grid, where the asymmetric offset is
   built into the object points as `x = 2·col + (row mod 2)`.
5. **Prove** the fit: reproject the theoretical grid through the recovered pose and
   overlay yellow crosshairs plus the warped plane boundary. If the crosshairs sit on
   the real dots, the pose is right. This is the check that makes the number
   trustworthy.
6. **Extract** pitch/yaw/roll from the rotation matrix via `Rodrigues`, with the
   gimbal-lock singular case handled explicitly, and draw the 3D axes.

[`reconstruction/Gen8.py`](../reconstruction/Gen8.py) is a one-off scratch file that
pretty-prints a recovered rotation matrix — kept because it records an actual solved
pose from the bench.

---

## Phase 3 · Acquisition

**[`acquisition/CamLive.py`](../acquisition/CamLive.py)** — live 1920×1080 feed from
the ELP camera. `s` saves the current frame, `q` quits. Deliberately minimal: the job
is to get an uncompressed, correctly-exposed frame of the fringe field on the fluid
while the field is on, and nothing else.

**[`reconstruction/GenMINI.py`](../reconstruction/GenMINI.py)** — a blur-and-threshold
scratch utility for cleaning a captured frame before it goes into the full pipeline.

---

## Phase 4 · Reconstruction — the Gen lineage

Each generation kept what worked and replaced what didn't. They are all preserved
because the differences between them *are* the engineering record.

| File | Approach | What changed |
|---|---|---|
| [`FerroTopGen2.py`](../reconstruction/FerroTopGen2.py) | Bidirectional contour tracing | First real engine. Sub-pixel peaks by log-parabola fit; traces each fringe left→right *and* right→left so the far side of a spike, hidden from one direction, is filled from the other. True ray-to-plane triangulation with explicit projector `R` and `T`. |
| [`FerroTopologyCOde.py`](../reconstruction/FerroTopologyCOde.py) | Vertical column unwrapping | Auto-calibrating scanner; unwraps phase down image columns instead of following contours. |
| [`Ferrotopogen3.py`](../reconstruction/Ferrotopogen3.py) | Graph-theoretic unwrapping | Priority-queue 2D unwrapping wavefront — visits pixels in order of confidence rather than in scan order, so unwrapping errors can't propagate along a whole row. |
| [`Gen4.py`](../reconstruction/Gen4.py) | Trigonometric height | Switches to displacement-from-baseline × the Phase 2 angle. Simple, fast, and the shape the rest of the lineage keeps. |
| [`Gen6.py`](../reconstruction/Gen6.py) | + glare suppression, + physical units | Illumination normalisation; output finally in millimetres via `px_per_mm`. Adds the topological metrics report — spike count, density, base area. |
| [`Gen7.py`](../reconstruction/Gen7.py) | + non-linear keystone baseline | The reference "floor" becomes a **quadratic** fit `y = ax² + bx + c` per fringe instead of a horizontal line, absorbing residual keystone curvature. Adds cubic inpainting of shadowed gaps and the Phase-2/Phase-3 diagnostic plots. |
| [`Gen7synthetic.py`](../reconstruction/Gen7synthetic.py) | Validation variant | Identical to Gen7 with the glare mask forced to zeros, so synthetic test fringes aren't deleted as specular highlights. |
| [`Gen8.1.py`](../reconstruction/Gen8.1.py) | + auto-crop, Chart Studio export | Auto-crops blown-out rows; publishes the render to a shareable link. Hardcodes `0173.json`. |
| [`Gen8.1test.py`](../reconstruction/Gen8.1test.py) | + circular ROI, full metrology report | Cookie-cutter circular mask at 85% radius removes the curved petri-dish wall, which otherwise contaminates the baseline fit. Adds surface area by gradient integration, peak statistics, and max-ascent angle. |
| [`Gen8nokeystonecorrection.py`](../reconstruction/Gen8nokeystonecorrection.py) | **Ablation control** | Gen8.1test with the quadratic baseline replaced by a flat horizontal one and the ROI crop removed. Exists to show how much the keystone correction is actually worth. |
| [`Gen8.2.py`](../reconstruction/Gen8.2.py) | Streamlined production run | Trimmed pipeline, seamless chrome-gradient renderer. The version used for routine reconstructions. |
| [`Gen9.py`](../reconstruction/Gen9.py) | + human-in-the-loop repair | Adds a "puzzle piece" composite: the centre of the frame, where glare is worst, is cut out and overlaid with the raw image so fringes can be traced by hand in red ink. The script then extracts the red channel, stitches the traced lines back into the binary mask, and continues. An honest answer to fringes that simply are not recoverable automatically. |

### The common stages

Every Gen from 4 onward runs the same five stages:

1. **Optics correction** — undistort with the Phase 1 matrices; auto-crop saturated rows.
2. **Fringe extraction** — glare masking (threshold + dilate), Gaussian blur, adaptive
   threshold, then a morphological close with a wide flat kernel to stitch fringes
   broken by specular dropouts.
3. **Triangulation** — per fringe, take the centre of mass of each connected run, fit
   the baseline (flat or quadratic), and convert `Δy` to height through the projector
   pitch angle and `px_per_mm`.
4. **Height field** — scatter the triangulated points onto a regular grid with
   `scipy.interpolate.griddata`, inpaint shadow gaps, and smooth lightly.
5. **Metrics + render** — spike statistics and an interactive Plotly surface.

### Reading the metrology report

The later generations print a report worth knowing how to read:

- **Active base area** — grid area above 1 mm, i.e. the fluid's actual footprint.
- **True 3D surface area** — `∫√(1 + (∂z/∂x)² + (∂z/∂y)²) dA`, the stretched membrane.
- **Surface membrane ratio** — the above divided by base area. A flat surface is 1.0×;
  the deviation is a scale-free measure of how spiked the surface is.
- **Peak std. deviation** — spread of spike heights, used as a **field uniformity**
  metric. A perfectly uniform field gives spikes of equal height; a gradient does not.
- **Maximum ascent angle** — steepest surface gradient, which tracks localised flux
  intensity.

---

## Phase 5 · Theory and analysis

| File | Role |
|---|---|
| [`theory/FerroSim.py`](../theory/FerroSim.py) | The continuum model. Derives bulk density and surface tension from the fluid's composition, computes the critical wavenumber and magnetisation, and renders the predicted hexagonal surface as three plane waves superimposed at 120°. |
| [`analysis/field_calibration.py`](../analysis/field_calibration.py) | Least-squares fit of B(I) through the origin, with residuals and coil resistance. Converts recorded drive currents into the field strengths quoted with each measurement. |
| [`analysis/spike_lattice.py`](../analysis/spike_lattice.py) | Decodes the base64 surface arrays back out of the bundled Plotly export, finds spikes as local maxima above 1.5 mm inside a 6% edge margin, and reports nearest-neighbour spacing. Regenerates both README figures. |

See [RESULTS.md](RESULTS.md) for what the numbers came to.
