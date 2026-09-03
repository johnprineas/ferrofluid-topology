# Ferrofluid Topology

**Structured-light 3D reconstruction of the Rosensweig instability in magnetised ferrofluid.**

Fringe-projection profilometry source code to measure the critical wavelength of the hexagonal spike lattice that forms when a ferrofluid surface is destabilised by a normal magnetic field.

<p align="center">
  <a href="results/ferrofluid_render_public.html"><strong>▶ Open the interactive 3D render</strong></a><br>
  <sub>HTML code available in link if doesn't open automatically</sub>
</p>

> **Research project · April – June 2026**
> Measured Rosensweig critical wavelength **8.3 ± 0.4 mm at 10.6 mT**, against **7.6 – 9.8 mm** predicted by a ferrohydrodynamic continuum model.

---

## The problem

Past a critical normal field, a ferrofluid surface becomes a hexagonal lattice of spikes. There is competition amongst magnetic energy, gravity and surface tension, so linear stability analysis fixes it (critical wavenumber $k_c$) at the inverse capillary length independently of the field that triggers it [1][2]:

$$k_c=\sqrt{\frac{\rho g}{\gamma}},\qquad \lambda_c=\frac{2\pi}{k_c}=2\pi\ell_c$$

The fluid is black and effectively specular, returning almost no diffuse signal for photogrammetry to correlate; it cannot be contacted without destroying the pattern under study; and the spikes are steep enough that most non-contact profilometry loses the flanks (deflected specular rays fall outside the numerical aperture of collection optics). Structured light circumvents all three by making the illumination itself the reference — project a known pattern onto the fluid, infer height from how it bends — which transposes a hard metrology problem into a calibration problem, and that is where most of this repository lives.

---

## How it works

- **`patterns/`** — generate projector targets at the DLP's native 640×360: checkerboard for lens intrinsics, asymmetric circle grid for projector pose (relative to camera), synthetic keystone warps at known angles for validation
- ↳ **`calibration/`, phase 1 — intrinsics** — Brown–Conrady $k_1,k_2,p_1,p_2$ tuned live against a projected checkerboard → `my_lens_metrics.json`
- ↳ **`calibration/`, phase 2 — extrinsics** — `findCirclesGrid(ASYMMETRIC)` → `solvePnP` → `Rodrigues` → projector pitch / yaw / roll
- ↳ **`acquisition/`** — cross-polarised global-shutter capture of the fringe field on the live fluid
- ↳ **`reconstruction/`, phase 3 — the Gen lineage** — glare suppression → sub-pixel centreline extraction → phase unwrapping for stripe indexing → baseline subtraction → triangulation → interpolated height field in millimetres
- ↳ **`theory/` + `analysis/`** — compare against the continuum model and the Hall-probe field calibration

Surface height is solved in closed form by intersecting the 1D line-of-sight ray from each camera pixel with the calibrated 2D light plane of its observed projector fringe, converting local phase shifts directly into vertical elevation:

$$z = \frac{p_0 \Delta\phi}{2\pi \tan\theta}$$

where:
- **$z$** — physical out-of-plane surface elevation (mm) relative to the flat baseline plane ($z = 0$)
- **$p_0$** — reference fringe pitch (mm), the spatial wavelength of one complete fringe period on the unperturbed reference plane
- **$\Delta\phi$** — local unwrapped phase shift (rad) between the deformed fluid surface and the calibrated flat reference, $\phi(x,y) - \phi_{\text{ref}}(x,y)$
- **$2\pi$** — angular normalization factor representing one complete fringe cycle in radians, converting $\Delta\phi$ to fractional fringe order displacement ($\Delta\phi / 2\pi$)
- **$\theta$** — triangulation baseline angle between the camera optical axis and the projector illumination axis
- **$\tan\theta$** — geometric projection factor relating out-of-plane elevation to lateral fringe displacement across the baseline plane ($\Delta x = z \tan\theta$)

## Gen code iterations and testing
<img width="2880" height="1800" alt="Screenshot From 2026-09-03 05-12-39" src="https://github.com/user-attachments/assets/5f6f6a4b-d138-4afd-8825-7ca1aac8e730" />
<img width="2880" height="1800" alt="Screenshot From 2026-09-03 05-12-44" src="https://github.com/user-attachments/assets/d68daef7-5121-4825-b635-152dd96bd005" />
---

## Final results

| Metric | Value |
|---|---|
| Measured critical wavelength | **8.3 ± 0.4 mm** at 10.6 mT |
| Continuum-model prediction | 7.6 – 9.8 mm |
| Bundled reconstruction | 7 spikes, peak 5.88 mm, NN spacing 7.42 ± 0.28 mm |

<p align="center">
<img width="2880" height="1800" alt="Screenshot From 2026-09-03 05-27-02" src="https://github.com/user-attachments/assets/587b92e0-b13a-4dd8-960e-463034070d96" />
<img width="2880" height="1800" alt="Screenshot From 2026-09-03 05-13-14" src="https://github.com/user-attachments/assets/2a34e46f-ce19-4ff5-8495-2f2500b3abe1" />

Spike counts, drive currents and Hall-probe readings are in [`data/`](data/). Full docs/RESULTS: [docs/RESULTS.md](docs/RESULTS.md).


The lattice is also hysteretic — the bifurcation is subcritical, so spikes appear at a higher field than the one at which they collapse. Up-sweep and down-sweep tables are kept separate in [`data/`](data/).

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


---

## Hardware

| Component | Part | Why |
|---|---|---|
| Projector | DLP LightCrafter 160CP EVM, 640×360 | Native resolution sets every pattern canvas; generating targets at any other size resamples the stripe edges and degrades the centreline fit |
| Camera | ELP AR0234, 1920×1200 global shutter, USB 3.0 | A rolling shutter shears a moving fluid surface |
| Optics | Crossed linear polarisers | Kills the specular glare off a mirror-black liquid |
| Field | Coil + Hall probe | 0 – 17.7 mT at the dish centre |


---

## License

[MIT](LICENSE).
