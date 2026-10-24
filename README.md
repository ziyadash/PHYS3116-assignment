# PHYS3116 Assignment: Accreted Milky Way Globular Clusters

**Option 1.** Classifying Galactic globular clusters as accreted (born in a dwarf galaxy
later absorbed by the Milky Way) or in-situ (born in the Milky Way itself), using stellar
populations and orbital dynamics.

## Results

**51 of 151 clusters classified as accreted, 34 per cent**, against the 35 per cent
reported by Belokurov & Kravtsov (2024). Per-cluster output in
`results/classifications.csv`.

**The model classified 14 clusters it had never seen and got all 14 right.** Those are
the recent discoveries absent from the 2010 Harris catalogue, so they never entered
training. That is the strongest result here, because unlike the cross-validated figures it
is measured on data the model never touched.

Full findings in `docs/results.md`, method in `docs/methodology.md`.

## Two models

| arm | inputs | n | AUC |
|---|---|---|---|
| dynamical: logistic regression | V_phi, Lz, eccentricity, log apocentre | 151 | 0.998 |
| stellar population: age-metallicity line | ages, [Fe/H] | 55 | 0.912 |

The split is deliberate. Ages exist for only 55 clusters while orbits reach 151, so no
single model can use everything the brief asks for without discarding most of the sample.

The 0.998 should not be read as a triumph. The reference labels are themselves defined
from orbital energy and angular momentum, so a model given orbital quantities largely
re-derives the boundary that produced them. The stellar arm uses no orbital information at
all, which makes its 0.912 the only non-circular result in the project.

## Data

| file | source | what it gives |
|---|---|---|
| `HarrisPartI.csv`, `HarrisPartIII.csv` | Harris catalogue (Moodle) | positions, distances, structural parameters |
| `vandenBerg_table2.csv` | VandenBerg et al. (2013) (Moodle) | ages and metallicities, 55 clusters |
| `Krause21.csv` | Krause et al. (Moodle) | masses, radii, compactness |
| `HarrisPartII.csv` | Harris catalogue, fetched | metallicities, 152 clusters |
| `Baumgardt_orbits.txt` | Baumgardt, Gaia-derived | proper motions and space velocities, 165 clusters |
| `BelokurovKravtsov24_tableD1.csv` | Belokurov & Kravtsov (2024), Appendix D | reference labels, transcribed from the PDF |

Anything not supplied with the assignment is documented in `data/SOURCES.md`, including
how it was retrieved, processed and checked.

## Running it

```
python src/main.py                   # full pipeline, writes figures/ and results/
python src/checks.py                 # sanity checks, mostly sign and frame tripwires
python src/fetch_harris_partii.py    # regenerate HarrisPartII.csv from source
```

Needs `pandas`, `numpy`, `matplotlib`.

## Layout

```
src/config.py                 paths, constants, figure saving
src/data_prep.py              load catalogues, reconcile names, merge, attach labels
src/exploratory.py            age-metallicity, spatial, structural and rotation figures
src/method_design.py          dynamical arm, ablation, held-out test, classification
src/rule_baseline.py          stellar population arm
src/checks.py                 sanity checks
src/fetch_harris_partii.py    regenerates one derived data file
docs/methodology.md           method, evaluation protocol, what survived and why
docs/results.md               findings as dot points
data/SOURCES.md               provenance for the non-supplied data
meetings/                     meeting minutes
```

The catalogues name the same cluster several different ways, so everything merges on a
canonical `PREFIX+NUMBER` key. `data_prep.py` holds the aliases and the reasoning.
