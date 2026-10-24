# Meeting Minutes 3
## Agenda
| Date | 17 October 2026 |
|--|--|
| Attendees | Ziyad |
| Purpose | Acquire orbital data, answer the rotation clause, and produce an actual classification result. |
| Next meeting | To be confirmed, pending the write-up |
| Author | Ziyad |

## Progress check
- Ziyad: acquiring the Gaia-derived orbit
  data unblocked the one spec clause that previously had no path at all, and the project
  now produces a result rather than only metrics.

## Orbital data
Added Baumgardt's orbit table, which is Gaia-derived and gives proper motions and full
space velocities for 165 clusters. Harris only ever had line-of-sight velocity, which is
why the rotation question had been stuck.

Derived three quantities from it: the azimuthal velocity V_phi, the vertical angular
momentum Lz, and orbital eccentricity.

## Rotation
In-situ clusters rotate at a median V_phi of +94 km/s. Accreted clusters sit at +8 km/s,
so they are effectively not rotating with the disc. The disc spin-up is visible: in-situ
clusters above [Fe/H] of -1.3 rotate at +131 km/s against +75 km/s for the metal-poor
ones. 45 of 151 clusters counter-rotate.

Accreted clusters are also on much more radial orbits, eccentricity 0.77 against 0.52,
and reach far further out, median apocentre 28.6 kpc against 4.0 kpc.

This reproduces the published rotation result from an independent derivation, so it
doubles as a check that our orbital quantities are correct.

## The circularity problem, confirmed
Adding orbital features pushes the classifier to AUC 0.999, misclassifying 3 of 151
clusters. This is not a success. The reference labels are defined from orbital energy and
angular momentum, so a model given orbital quantities is re-deriving the boundary that
produced the labels.

The figure worth reporting is 0.929 from stellar populations alone, which uses no orbital
information and is therefore a genuine claim about ages and metallicities rather than a
restatement of orbits.

## Result
51 of 151 clusters classified as accreted, 34 per cent, against the 35 per cent in the
reference paper. We disagree with their published label on two clusters only, NGC 5053
and Pal 5, both borderline. Per-cluster output is written to `results/classifications.csv`.

On the clusters the reference authors themselves flag as uncertain, our stellar-population
model sides with their chemistry against their orbital label on three of the five we can
see. That is the closest thing this project has to an original contribution, since every
Milky Way cluster is already classified by someone.

Everything code-wise is basically done, and I'm going to be working on the slides now.
The presentation is something I'll record close to the due date, but I'm happy that the
bulk of the work (the code, and gathering the results) is completed.

## Housekeeping
Wrote a set of sanity checks covering name matching, merge integrity, coordinate frames,
sign conventions and fold handling. The sign checks matter most: a flipped V_phi would
have produced entirely reasonable plots and wrong conclusions. These later caught a name
alias bug where Baumgardt writes `Djor_1` against everyone else's `Djorg 1`, which had
been quietly dropping two clusters.

Trimmed the reported metrics from five to three after checking that balanced accuracy,
recall and precision are all near-perfectly predictable from each other, since they come
from the same confusion counts.
