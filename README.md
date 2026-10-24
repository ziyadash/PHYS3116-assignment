# PHYS3116 Assignment: Accreted Milky Way Globular Clusters

**Option 1.** Classifying Galactic globular clusters as accreted (born in a dwarf galaxy
later absorbed by the Milky Way) or in-situ (born in the Milky Way itself), using stellar
populations and orbital dynamics.

## Result

51 of 151 clusters classified as accreted, 34 per cent, against the 35 per cent reported
by Belokurov & Kravtsov (2024). Full per-cluster output in `results/classifications.csv`.

Age and metallicity alone separate the two populations at AUC 0.929, with no orbital
information used. Orbital features reach 0.999, but that is close to circular: the
reference labels are themselves defined from orbital energy and angular momentum. See
`docs/results.md`.

## Data

| file | source | what it gives |
|---|---|---|
| `HarrisPartI.csv`, `HarrisPartIII.csv` | Harris catalogue | positions, distances, structural parameters |
| `vandenBerg_table2.csv` | VandenBerg et al. (2013) | ages and metallicities, 55 clusters |
| `Krause21.csv` | Krause et al. | masses, radii, compactness |
| `Baumgardt_orbits.txt` | Baumgardt (Gaia-derived) | proper motions and space velocities, 165 clusters |
| `BelokurovKravtsov24_tableD1.csv` | Belokurov & Kravtsov (2024), Appendix D | in-situ/accreted labels, transcribed by hand from the PDF |

## Running it

```
python src/main.py      # full pipeline, writes figures/ and results/
python src/checks.py    # sanity checks, mostly sign and coordinate-frame tripwires
```

Needs `pandas`, `numpy`, `matplotlib`.

## Layout

```
src/config.py          paths, constants, figure saving
src/data_prep.py       load four catalogues, reconcile names, merge, attach labels
src/exploratory.py     age-metallicity, spatial, structural and rotation figures
src/method_design.py   logistic regression on a designed feature vector
src/rule_baseline.py   two-parameter rule, kept as the comparison
src/checks.py          sanity checks
docs/methodology.md    method, evaluation protocol, why each metric and feature survived
docs/results.md        findings as dot points
meetings/              meeting minutes
```

The three catalogues name the same cluster three different ways, so everything is merged
on a canonical `PREFIX+NUMBER` key. `data_prep.py` has the details and the aliases.
