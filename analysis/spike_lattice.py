"""Recover the height field from a Plotly render and measure the spike lattice.

`results/ferrofluid_render_public.html` is a self-contained Plotly export of a
reconstruction: the surface arrays are embedded in it as base64 float64 blobs.
This script pulls them back out, so the figures in the README are reproducible
from the artefact that ships with the repo rather than from a lost session.

It then locates spikes as local maxima of the height field and reports the
nearest-neighbour spacing, which for a hexagonal Rosensweig lattice is the
observable that maps onto the critical wavelength.

Usage:
    python analysis/spike_lattice.py
    python analysis/spike_lattice.py --figures assets/figures
"""

import argparse
import base64
import json
import os
import re

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

# Peaks must clear this height to count as a spike rather than floor texture.
MIN_SPIKE_HEIGHT_MM = 1.5
# Local-maximum window, in grid cells.
PEAK_WINDOW = 25
# Fraction of the grid trimmed from each edge; the dish wall and the ends of the
# fringe traces sit here and produce spurious maxima.
EDGE_MARGIN_FRAC = 0.06


def load_surface(html_path):
    """Return (X, Y, Z) mm arrays from the Plotly surface trace embedded in an export."""
    with open(html_path, encoding="utf-8", errors="replace") as f:
        html = f.read()

    call = re.search(r'Plotly\.newPlot\(\s*"[0-9a-f\-]+",\s*', html)
    if call is None:
        raise ValueError(f"no Plotly.newPlot payload found in {html_path}")

    traces, _ = json.JSONDecoder().raw_decode(html[call.end():])
    surfaces = [t for t in traces if t.get("type") == "surface"]
    if not surfaces:
        raise ValueError(f"no surface trace found in {html_path}")
    trace = surfaces[0]

    def to_array(field):
        # Plotly v3 writes large arrays as {'dtype', 'bdata', 'shape'}.
        if isinstance(field, dict):
            flat = np.frombuffer(base64.b64decode(field["bdata"]), dtype=np.dtype(field["dtype"]))
            return flat.reshape([int(n) for n in field["shape"].split(",")])
        return np.asarray(field, dtype=float)

    return to_array(trace["x"]), to_array(trace["y"]), to_array(trace["z"])


def find_spikes(X, Y, Z):
    """Return (x_mm, y_mm, height_mm) of spike peaks, tallest first."""
    rows = Z.shape[0]
    margin = int(rows * EDGE_MARGIN_FRAC)
    interior = np.zeros_like(Z, dtype=bool)
    interior[margin:rows - margin, margin:Z.shape[1] - margin] = True

    neighbourhood = ndimage.maximum_filter(Z, size=PEAK_WINDOW)
    peaks = (Z == neighbourhood) & (Z > MIN_SPIKE_HEIGHT_MM) & interior

    row_idx, col_idx = np.where(peaks)
    x, y, z = X[row_idx, col_idx], Y[row_idx, col_idx], Z[row_idx, col_idx]
    order = np.argsort(-z)
    return x[order], y[order], z[order]


def nearest_neighbour_spacing(x, y):
    """Distance from each spike to its closest neighbour, in mm."""
    points = np.column_stack([x, y])
    distances, _ = cKDTree(points).query(points, k=2)
    return distances[:, 1]


def _dark_axes(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")


def write_figures(X, Y, Z, x, y, spacing, outdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    sem = spacing.std(ddof=1) / np.sqrt(len(spacing))

    fig = plt.figure(figsize=(12, 7), facecolor="#0d1117")
    ax = fig.add_axes([-0.02, -0.10, 0.86, 1.16], projection="3d", facecolor="#0d1117")
    step = 2
    surf = ax.plot_surface(X[::step, ::step], Y[::step, ::step], Z[::step, ::step],
                           cmap="inferno", linewidth=0, antialiased=True)
    ax.set_xlabel("X (mm)", color="#c9d1d9", labelpad=10)
    ax.set_ylabel("Y (mm)", color="#c9d1d9", labelpad=10)
    ax.set_zlabel("Height (mm)", color="#c9d1d9", labelpad=6)
    ax.tick_params(colors="#8b949e", pad=2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#0d1117")
        axis.pane.set_edgecolor("#30363d")
    ax.set_box_aspect((X.max() - X.min(), Y.max() - Y.min(), 13))
    ax.view_init(elev=34, azim=-125)
    ax.set_zlim(0, max(6.0, float(Z.max())))
    cax = fig.add_axes([0.87, 0.26, 0.018, 0.46])
    bar = fig.colorbar(surf, cax=cax)
    bar.set_label("True height (mm)", color="#c9d1d9")
    bar.ax.tick_params(colors="#8b949e")
    bar.outline.set_edgecolor("#30363d")
    fig.text(0.44, 0.945,
             f"Reconstructed Rosensweig topography  ·  "
             f"{X.max() - X.min():.1f} × {Y.max() - Y.min():.1f} mm field of view",
             color="#e6edf3", fontsize=14, ha="center")
    fig.savefig(os.path.join(outdir, "reconstruction_3d.png"), dpi=130, facecolor="#0d1117")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 7), facecolor="#0d1117")
    _dark_axes(ax)
    im = ax.imshow(Z, cmap="inferno", origin="lower",
                   extent=[X.min(), X.max(), Y.min(), Y.max()])
    ax.scatter(x, y, s=170, facecolors="none", edgecolors="#58a6ff", linewidths=1.8,
               label=f"{len(x)} spikes  ·  NN spacing {spacing.mean():.1f} ± {sem:.1f} mm")
    ax.set_xlabel("X (mm)", color="#c9d1d9")
    ax.set_ylabel("Y (mm)", color="#c9d1d9")
    ax.set_title("Height field · detected spike lattice", color="#e6edf3", fontsize=14)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9",
              loc="upper right", fontsize=9)
    bar = fig.colorbar(im, ax=ax, shrink=0.85)
    bar.set_label("Height (mm)", color="#c9d1d9")
    bar.ax.tick_params(colors="#8b949e")
    bar.outline.set_edgecolor("#30363d")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "heightmap_peaks.png"), dpi=135, facecolor="#0d1117")
    plt.close(fig)

    print(f"\nWrote reconstruction_3d.png and heightmap_peaks.png to {outdir}")


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", default=os.path.join(here, "results", "ferrofluid_render_public.html"))
    parser.add_argument("--figures", default=None, help="directory to write the README figures into")
    args = parser.parse_args()

    X, Y, Z = load_surface(args.render)
    x, y, heights = find_spikes(X, Y, Z)
    spacing = nearest_neighbour_spacing(x, y)
    sem = spacing.std(ddof=1) / np.sqrt(len(spacing))

    cell = ((X.max() - X.min()) / X.shape[1]) * ((Y.max() - Y.min()) / Y.shape[0])
    print(f"Field of view         : {X.max() - X.min():.1f} x {Y.max() - Y.min():.1f} mm")
    print(f"Grid                  : {Z.shape[0]} x {Z.shape[1]}")
    print(f"Spikes detected       : {len(heights)}  (> {MIN_SPIKE_HEIGHT_MM} mm)")
    print(f"Peak height           : max {heights.max():.2f} mm, mean {heights.mean():.2f} mm, "
          f"sd {heights.std(ddof=1):.2f} mm")
    print(f"NN spacing            : {spacing.mean():.2f} +/- {sem:.2f} mm "
          f"(median {np.median(spacing):.2f}, sd {spacing.std(ddof=1):.2f})")
    print(f"Fluid volume above 0  : {float(Z.sum()) * cell:.1f} mm^3")
    print("\nSpike peaks (x_mm, y_mm, height_mm):")
    for xi, yi, zi in zip(x, y, heights):
        print(f"  {xi:6.2f}  {yi:6.2f}  {zi:5.2f}")

    if args.figures:
        write_figures(X, Y, Z, x, y, spacing, args.figures)


if __name__ == "__main__":
    main()
