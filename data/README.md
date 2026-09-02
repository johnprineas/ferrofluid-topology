# Experimental data

Transcribed from the original lab spreadsheets. The source screenshots are kept
alongside the CSVs so every value can be checked against what was actually recorded.

| File | Contents |
|---|---|
| `electromagnet_calibration.csv` | 15 Hall-probe points, coil drive vs. flux density at the dish centre |
| `spike_count_up_sweep.csv` | Spike count vs. drive, increasing field (runs A and C) |
| `spike_count_down_sweep.csv` | Spike count vs. drive, decreasing field (run B) |
| `spike_breakdown_transitions.csv` | Drive point at which the lattice collapsed from one count to the next (repeat runs D and E) |
| `source_electromagnet_calibration.png` | Source spreadsheet for the calibration |
| `source_spike_count_tables.png` | Source spreadsheet for all spike-count tables |

Each CSV carries `#` comment lines at the top describing its columns; both
`analysis/*.py` readers skip them.

## Conventions

**`run`** distinguishes independent sweeps. A, C are up-sweeps; B, D, E are
down-sweeps. Up and down are kept in separate files because the lattice is hysteretic
— spikes appear at a higher field than the one at which they collapse — so pooling
them would average across a real physical effect.

**`spikes`** is recorded as it was observed, and is not always an integer:

- `Primary` — the first spike at instability onset, before a stable count resolves.
- `4->3` — the count was collapsing between the two values at that drive point.
- `8->7->4` — a multi-step collapse observed at one drive point.

Parse this column as a string. The counts are visual observations from the live feed,
not automatic detections; the automatic spike counter in the reconstruction scripts is
a separate measurement of a separate quantity.

**`approx`** in `spike_breakdown_transitions.csv` flags rows the source marked as
approximate; the source also left one current cell blank, kept empty here rather than
guessed.

**Fields** are not stored in the spike-count files except where a Hall probe was read
directly (run C's `B_centre_mT`). Everywhere else, convert from `current_mA` using the
fit in `analysis/field_calibration.py` — `B = 54.37 mT/A × I`.

## Reading them

```python
import pandas as pd

cal = pd.read_csv("data/electromagnet_calibration.csv", comment="#")
up  = pd.read_csv("data/spike_count_up_sweep.csv", comment="#")

up["B_mT"] = up["current_mA"] * 0.05437   # mT per mA
```
