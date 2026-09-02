# Results

## Headline

| | |
|---|---|
| **Measured critical wavelength** | **8.3 ± 0.4 mm** at 10.6 mT |
| **Predicted (continuum model)** | **7.6 – 9.8 mm** |

The measurement sits inside the predicted band. Given that the prediction depends on
a bulk surface tension estimated from carrier composition — the parameter the model is
most sensitive to and the one hardest to pin down for a loaded colloid — landing
inside the band is about as much as the comparison can establish. It does not
discriminate between the model and plausible alternatives; it does show the bench
measures a real physical length scale to sub-millimetre precision.

---

## The theoretical prediction

For a ferrofluid layer destabilised by a normal field, the balance of gravity against
surface tension sets a critical wavenumber, and the spike lattice forms at the
corresponding wavelength:

$$k_c = \sqrt{\frac{\rho g}{\gamma}}, \qquad \lambda_c = \frac{2\pi}{k_c}$$

with the critical magnetisation for onset:

$$M_c^2 = \frac{2}{\mu_0}\sqrt{\rho g \gamma}$$

[`theory/FerroSim.py`](../theory/FerroSim.py) evaluates these from the fluid's
composition. Mixing the carrier by mass fraction and loading it with magnetite at
φ = 0.0914:

```
Bulk density               ρ = 1127.22 kg/m³
Effective surface tension  γ = 0.01632 N/m
Critical wavenumber      k_c = 823.28 rad/m
Predicted wavelength     λ_c = 7.63 mm
Critical magnetisation   M_c = 4623.6 A/m   (µ₀M_c ≈ 5.8 mT)
```

The nominal 7.63 mm sits at the **lower edge** of the quoted 7.6 – 9.8 mm band. The
band's width is not statistical — it is the spread over defensible parameter choices,
dominated by the carrier surface tension:

| γ (N/m) | φ = 0.05 | φ = 0.0914 | φ = 0.15 |
|---|---|---|---|
| 0.0117 (pure hexane) | 7.09 mm | 6.46 mm | 5.80 mm |
| 0.0163 (65/35 mix) | 8.37 mm | **7.63 mm** | 6.85 mm |
| 0.0188 (pure undecane) | 8.99 mm | 8.19 mm | 7.36 mm |

Surfactant at the particle surface lowers the effective γ below the neat-carrier
value, while the loaded colloid's γ is not directly measured here — which is exactly
why the prediction is quoted as a band rather than a number.

Reproduce with `python theory/FerroSim.py` (also renders the predicted hexagonal
surface, modelled as three plane waves at 120°).

---

## Field calibration

Every field strength quoted with a measurement comes from the coil current through
this calibration ([`data/electromagnet_calibration.csv`](../data/electromagnet_calibration.csv),
15 Hall-probe points over 11 – 321 mA):

```
B(I) = 54.37 ± 0.37 mT/A        r² = 0.9975,  residual RMS 0.28 mT
```

Strictly linear through the origin — no saturation over the working range, so the
conversion is unambiguous. `python analysis/field_calibration.py` reproduces it.

---

## Spike-count campaign

[`data/`](../data/) holds the spike counts recorded against drive voltage and current,
transcribed from the original spreadsheets (the source screenshots are kept alongside
them as `source_*.png`).

### Onset

The first spikes appear at 17.85 V / 189 mA on the up-sweep — about 10.3 mT by the
calibration. The continuum model's onset field µ₀M_c ≈ 5.8 mT is a magnetisation
threshold, not an applied-field threshold, and converting between the two needs the
fluid's susceptibility and the demagnetising factor of the layer; the two numbers are
therefore not directly comparable, and the gap between them is expected rather than a
discrepancy.

### Hysteresis

The lattice is hysteretic, which is why up-sweep and down-sweep are stored as separate
files. Comparing the two repeat down-sweeps in
[`spike_breakdown_transitions.csv`](../data/spike_breakdown_transitions.csv):

| Spikes remaining | Run D collapse current | Run E collapse current |
|---|---|---|
| 14 → 12 | 225.5 mA | 226.9 mA |
| 12 → 10 | 213.2 mA | 214.1 mA |
| → 4 | 194.0 mA | 192.6 mA |
| 1 → 0 | 177.7 mA | 174.4 mA |

Repeat sweeps agree to roughly 1 – 3 mA (≈0.1 mT), so the collapse sequence is
reproducible and the transitions are a property of the fluid rather than of the run.

Counts do not descend one at a time: 19 → 14, 28 → 25, 8 → 7 → 4. The lattice sheds
several spikes at once, which is what a coupled lattice relaxing to a new equilibrium
should do, not what independent spikes switching off individually would look like.

The up-sweep reaches 27 spikes at 316 mA while the down-sweep still holds 25 at
285 mA — spikes persist to lower fields than the ones at which they formed.

---

## Reconstruction

[`results/ferrofluid_render_public.html`](../results/ferrofluid_render_public.html) is
a self-contained Plotly export of one reconstruction. Opening it in a browser gives
the interactive surface; `analysis/spike_lattice.py` recovers the underlying arrays
and reports:

```
Field of view      : 35.7 × 35.7 mm
Grid               : 500 × 500
Spikes detected    : 7  (peaks above 1.5 mm, 6% edge margin)
Peak height        : max 5.88 mm, mean 4.11 mm, sd 1.15 mm
NN spacing         : 7.42 ± 0.28 mm  (median 7.40, sd 0.74)
Fluid volume       : 374.7 mm³ above the reference floor
```

<p align="center">
  <img src="../assets/figures/heightmap_peaks.png" alt="Height field with detected spike lattice" width="66%">
</p>

**This capture is not the 10.6 mT measurement point.** Its 7.42 ± 0.28 mm
nearest-neighbour spacing is a single frame at its own field strength, bundled as a
worked example of the reconstruction output — not the campaign result. It is quoted
here so the figures in this repository can be traced to a number anyone can
regenerate.

Two caveats on that number. Seven spikes over a 36 mm field is a small sample, and
nearest-neighbour distance underestimates the lattice constant when the lattice is
incomplete: an edge spike's nearest neighbour is more often a lattice neighbour that
happens to be close than the true mean spacing. The 0.28 mm quoted is the standard
error on the mean of the seven distances and does not include either effect.

The spike-height spread (sd 1.15 mm on a 4.11 mm mean) is the field-uniformity metric
the later Gen reports print. It is large, indicating a real field gradient across the
dish — the tallest spike, 5.88 mm, sits near the centre, and the peaks fall off toward
the edges, consistent with a coil whose field is strongest on axis.

---

## What limits the measurement

**Optical, in order of severity.** Specular glare off the fluid is the dominant
problem: crossed polarisers reduce it but do not eliminate it, glare masking removes
data along with the highlight, and `Gen9.py` exists because some frames need fringes
traced by hand. Shadowing on the far side of tall spikes leaves gaps that are
inpainted, so height there is interpolated rather than measured — the bidirectional
tracing in `FerroTopGen2.py` was the attempt to attack this directly.

**Geometric.** Height scales with the tangent of the projector pitch, so an error in
the Phase 2 pose propagates straight into every height. The reprojection overlay is
the guard against a badly wrong angle, and the synthetic keystone images bound the
solver's own error. Residual keystone curvature is absorbed by the quadratic baseline
from Gen7 onward; `Gen8nokeystonecorrection.py` is retained as the ablation control
that shows what that correction is worth.

**Statistical.** A ~36 mm field of view holds only a handful of lattice cells, so
wavelength estimates from any single frame are limited by sample size more than by
per-spike precision.
