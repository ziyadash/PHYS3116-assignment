# Data sources

Supplied with the assignment, downloaded from Moodle:
`HarrisPartI.csv`, `HarrisPartIII.csv`, `Krause21.csv`, `vandenBerg_table2.csv`

Everything else was sourced during the project.

## HarrisPartII.csv
Metallicities for 152 of the 157 Harris clusters. Part II was missing from the Moodle
files, which held `[Fe/H]` coverage down to the 55 clusters VandenBerg covers. Used to
test whether metallicity adds anything to the dynamical model once orbits are present; it
does not, so the model no longer uses it, but it stays as context in the output table.

Parsed out of the full Harris catalogue at `physics.mcmaster.ca/~harris/mwgc.dat`, the
same catalogue the Moodle files come from. Regenerate with
`python src/fetch_harris_partii.py`. Five clusters have no published `[Fe/H]`.

## Baumgardt_orbits.txt
Gaia-derived proper motions and space velocities for 165 clusters. This is the dataset the
dynamical model runs on. Harris only has line-of-sight velocities, so before this the
rotation question could not be answered at all and the model was limited to 55 clusters.

`V_phi`, `Lz` and eccentricity are derived from it in `load_orbits`, not taken from the
file. Stored exactly as downloaded from
`people.smp.uq.edu.au/HolgerBaumgardt/globular/orbits_table.txt`.

## BelokurovKravtsov24_tableD1.csv
In-situ and accreted classifications for 165 clusters. These are the reference labels both
models are scored against, and the source of the 14-cluster held-out test. Their comments
column also flags the clusters they consider possibly misclassified, which is what the
stellar model is compared against.

Transcribed by hand from Appendix D of Belokurov & Kravtsov (2024), MNRAS 528, 3198, since
it is not available machine-readable from VizieR or CDS. Verified against the PDF text
layer, which caught one error; the file reproduces the paper's stated counts exactly.

All retrieved October 2026.
