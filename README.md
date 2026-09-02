# Ferrofluid Topology

**Structured-light 3D reconstruction of the Rosensweig instability in magnetised ferrofluid.**

Fringe-projection profilometry, the calibration chain that makes it defensible, and the
measurement it was built to make: the critical wavelength of the hexagonal spike lattice
that forms when a ferrofluid surface is destabilised by a normal magnetic field.

<p align="center">
  <a href="results/ferrofluid_render_public.html"><strong>▶ Open the interactive 3D render</strong></a><br>
  <sub>Self-contained Plotly export — rotate, zoom and inspect the reconstructed surface. Works offline.</sub>
</p>

> **Research project · April – June 2026**
> Measured Rosensweig critical wavelength **8.3 ± 0.4 mm at 10.6 mT**, against
> **7.6 – 9.8 mm** predicted by a ferrohydrodynamic continuum model.

---

## The problem

Past a critical normal field, a ferrofluid surface buckles into a hexagonal lattice of
spikes. The scale it selects is set by the competition amongst magnetic energy, gravity
and surface tension alone, so linear stability analysis fixes it at the inverse
capillary length independently of the field that triggers it [1][2]:

$$k_c=\sqrt{\frac{\rho g}{\gamma}},\qquad \lambda_c=\frac{2\pi}{k_c}=2\pi\ell_c$$

Measuring it is awkward. The fluid is black and effectively specular, returning almost
no diffuse signal for photogrammetry to correlate; it cannot be contacted without
destroying the pattern under study; and the spikes are steep enough that most
non-contact profilometry loses the flanks. Structured light circumvents all three by
making the illumination itself the reference — project a known pattern onto the fluid,
infer height from how it bends — which transposes a hard metrology problem into a
calibration problem, and that is where most of this repository lives.

---

## How it works

- **`patterns/`** — generate projector targets at the DLP's native 640×360: checkerboard
  for lens intrinsics, asymmetric circle grid for projector pose, synthetic keystone
  warps at known angles for validation
- ↳ **`calibration/`, phase 1 — intrinsics** — Brown–Conrady $k_1,k_2,p_1,p_2$ tuned live
  against a projected checkerboard → `my_lens_metrics.json`
- ↳ **`calibration/`, phase 2 — extrinsics** — `findCirclesGrid(ASYMMETRIC)` → `solvePnP`
  → `Rodrigues` → projector pitch / yaw / roll
- ↳ **`acquisition/`** — cross-polarised global-shutter capture of the fringe field on
  the live fluid
- ↳ **`reconstruction/`, phase 3 — the Gen lineage** — glare suppression → sub-pixel
  centreline extraction → phase unwrapping for stripe indexing → baseline subtraction →
  triangulation → interpolated height field in millimetres
- ↳ **`theory/` + `analysis/`** — compare against the continuum model and the Hall-probe
  field calibration

Each projected stripe is a plane, each camera pixel a ray; their intersection is a 3D
point. For a displacement $z$ seen across a triangulation angle $\theta$, a fringe of
pitch $p_0$ on the calibrated reference plane shifts by $\Delta x=z\tan\theta$, giving
$z=p_0\Delta\phi/2\pi\tan\theta$ and a depth sensitivity of $p_0/2\pi\tan\theta$. That
figure dictates every hardware choice: shallow $\theta$ buys field of view at the cost
of resolution, steep $\theta$ buys resolution at the cost of occluding the far flank of
every spike.

### Why each step is there

**Intrinsics first.** Barrel distortion reaches a few percent at the frame edge,
comparable to the spike heights being measured, so an uncorrected lens model deposits a
radial bias indistinguishable from genuine surface curvature.
`calibration/Gen1checkerboarddistortion.py` walks the coefficients on the arrow keys
with the undistorted image updating live, since the residual that matters is the one
visible at the edge and not the one minimised in aggregate. It writes the
`camera_matrix`, `distortion_coefficients`, `optimal_matrix` and `mm_per_pixel` every
later stage reads; schema in
[`calibration/example_lens_metrics.json`](calibration/example_lens_metrics.json).

**An asymmetric grid for pose.** A symmetric dot grid admits a 180° ambiguity and would
let the solver flip the sign of the projector tilt silently, propagating into
$\tan\theta$ and inverting the whole height field; an asymmetric one does not.
`calibration/Gen1angulardetermination.py` reprojects the theoretical grid back onto the
image so the fit is visually falsifiable rather than a number to be trusted on
assertion. Detection on a glowing pattern reflected off a dark liquid is the fragile
step, so it retries across the green, red, blue, high-contrast and plain grayscale
channels before conceding, and reports which succeeded.

**Synthetic validation.** `patterns/circulardistortionreverserengineering.py` warps a
pristine grid through a homography built from *known* angles; if the solver cannot
recover an angle it was handed, the bench is not the problem.
`reconstruction/Gen7synthetic.py` is the matching variant with glare masking disabled,
so synthetic fringes are not destroyed by a filter tuned for real ones.

**Acquisition.** A global shutter is mandatory, not preferable — a rolling shutter
shears a moving surface into an apparent tilt indistinguishable from real topography.
Crossed polarisers suppress the polarisation-preserving specular return that would
otherwise saturate at the glint and artificially broaden the fringe maxima.

**Height from fringes.** The work is getting a clean centreline out of a blurred, glared
stripe: the peak is fitted to sub-pixel precision by a parabola through the log of the
intensity profile, a Gaussian fit in closed form requiring no iteration, and stripes are
indexed by unwrapping outward from a calibrated flat reference so a spike displacing a
fringe by more than one period does not alias into its neighbour.

---

## Results

| | |
|---|---|
| Measured critical wavelength | **8.3 ± 0.4 mm** at 10.6 mT |
| Continuum-model prediction | 7.6 – 9.8 mm |
| Field calibration | 54.37 ± 0.37 mT/A, r² = 0.9975 |
| Bundled reconstruction | 7 spikes, peak 5.88 mm, NN spacing 7.42 ± 0.28 mm |

<p align="center">
  <img src="assets/figures/heightmap_peaks.png" alt="Height field with detected spike lattice" width="62%">
</p>

Both figures are regenerated from
[`results/ferrofluid_render_public.html`](results/ferrofluid_render_public.html) by
`analysis/spike_lattice.py`, so they stay tied to the data rather than to a lost session.
Spike counts, drive currents and Hall-probe readings are in [`data/`](data/). Full
write-up: **[docs/RESULTS.md](docs/RESULTS.md)**.

Four caveats travel with those numbers:

- **The prediction band is broad because $\gamma$ is not independently known here.**
  7.6 – 9.8 mm is the image of $\gamma\approx17$–$29\ \mathrm{mN\,m^{-1}}$ propagated
  through $\lambda_c=2\pi\sqrt{\gamma/\rho g}$, so agreement is a consistency check, not
  a discriminating test.
- **Nearest-neighbour spacing is not $\lambda_c$.** A hexagonal lattice separates its
  spikes by $a_{\rm NN}=(2/\sqrt3)\lambda\approx1.155\lambda$, and the bundled frame is
  supercritical, where the selected wavenumber grows with field [6]. Both effects push
  the same way; `analysis/spike_lattice.py` reports the two quantities separately so
  they cannot be silently merged.
- **The field is an applied field.** The stability criterion is stated on the
  magnetisation *inside* the fluid, but the Hall probe reads the dish centre; for a flat
  layer magnetised normal to its plane the demagnetising factor approaches unity. This
  is the dominant systematic.
- **Peak height is a lower bound.** Fringe order survives only while $|\nabla z|\lesssim
  \cot\theta$, and a specular flank tilted by $\alpha$ deflects the reflected lobe by
  $2\alpha$ out of the camera aperture before even that limit is reached. A 5.88 mm peak
  on a 7.42 mm lattice implies flank gradients of order 1.6, near the ceiling. Lateral
  peak position survives partial flank loss, so the wavelength statistic is far more
  robust than the amplitude one.

The lattice is also **hysteretic** — the bifurcation is subcritical, so spikes appear at
a higher field than the one at which they collapse. Up-sweep and down-sweep tables are
kept separate in [`data/`](data/) and must not be pooled.

---

## Repository map

| Path | What's in it |
|---|---|
| [`patterns/`](patterns/) | Projector target generation and the synthetic keystone validator |
| [`calibration/`](calibration/) | Phase 1 lens intrinsics, phase 2 projector pose, example metrics JSON |
| [`acquisition/`](acquisition/) | Live camera feed and frame capture |
| [`reconstruction/`](reconstruction/) | The Gen2 → Gen9 reconstruction lineage |
| [`theory/`](theory/) | Ferrohydrodynamic continuum model and predicted topology |
| [`analysis/`](analysis/) | Field calibration fit, spike-lattice statistics, figure generation |
| [`data/`](data/) | Experimental CSVs + the source spreadsheet screenshots |
| [`results/`](results/) | Interactive 3D render of a reconstruction |
| [`docs/`](docs/) | [Pipeline](docs/PIPELINE.md) · [Hardware](docs/HARDWARE.md) · [Results](docs/RESULTS.md) |

Filenames are kept as they were written during the project. They are not tidy, but they
are the actual lineage, and the tables in [docs/PIPELINE.md](docs/PIPELINE.md) map every
one of them to its role.

---

## Quickstart

```bash
git clone git@github.com:johnprineas/ferrofluid-topology.git
cd ferrofluid-topology
pip install -r requirements.txt
```

Nothing below needs the hardware:

```bash
python theory/FerroSim.py                          # predicted critical wavelength
python analysis/field_calibration.py --plot assets/figures/field_calibration.png
python analysis/spike_lattice.py --figures assets/figures
```

With a projector and camera attached the order is fixed by the dependency chain, since
each stage consumes the artefact written by the one before it:

```bash
python patterns/checkerboardgenerator.py          # 1. make targets
python patterns/circlegeneratorforangle.py
python acquisition/CamLive.py                     # 2. capture; 's' saves, 'q' quits
python calibration/Gen1checkerboarddistortion.py  # 3. tune intrinsics  -> metrics JSON
python calibration/Gen1angulardetermination.py    # 4. solve projector pose -> pitch
python reconstruction/Gen8.2.py                   # 5. reconstruct -> 3D height field
```

Steps 3–5 are interactive: they prompt for the metrics JSON, the projector pitch angle
and the pixels-per-millimetre scale, then open OpenCV and Plotly windows. `Gen8.1.py`
and `Gen8.2.py` expect a metrics file named `0173.json` in the working directory — copy
`calibration/example_lens_metrics.json` to that name, or use `Gen9.py`, which prompts.

**Requires:** Python 3.9+, and a desktop session — the calibration and reconstruction
tools open OpenCV GUI windows and use arrow-key HUDs.

---

## Hardware

| Component | Part | Why |
|---|---|---|
| Projector | DLP LightCrafter 160CP EVM, 640×360 | Native resolution sets every pattern canvas; generating targets at any other size resamples the stripe edges and degrades the centreline fit |
| Camera | ELP AR0234, 1920×1200 global shutter, USB 3.0 | A rolling shutter shears a moving fluid surface |
| Optics | Crossed linear polarisers | Kills the specular glare off a mirror-black liquid |
| Field | Coil + Hall probe | 0 – 17.7 mT at the dish centre, linear at 54.37 ± 0.37 mT/A, calibrated in [`data/`](data/) |

## References

1. R. E. Rosensweig, *Ferrohydrodynamics*. Cambridge University Press, 1985.
2. M. D. Cowley and R. E. Rosensweig, "The interfacial stability of a ferromagnetic fluid," *J. Fluid Mech.* **30**, 671–688 (1967).
3. D. C. Brown, "Decentering distortion of lenses," *Photogrammetric Engineering* **32**(3), 444–462 (1966).
4. K. Itoh, "Analysis of the phase unwrapping algorithm," *Appl. Opt.* **21**(14), 2470 (1982).
5. C. Gollwitzer *et al.*, "The surface topography of a magnetic fluid: a quantitative comparison between experiment and numerical simulation," *J. Fluid Mech.* **571**, 455–474 (2007).
6. B. Abou, J.-E. Wesfreid and S. Roux, "The normal field instability in ferrofluids: hexagon–square transition mechanism and wavenumber selection," *J. Fluid Mech.* **416**, 217–237 (2000).
7. S. Zhang, "High-speed 3D shape measurement with structured light methods: a review," *Opt. Lasers Eng.* **106**, 119–131 (2018).

---

## License

[MIT](LICENSE).
