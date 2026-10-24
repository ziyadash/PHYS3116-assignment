# Results

Brief notes, not a report. Everything here is reproduced by `python src/main.py`.
Per-cluster output is in `results/classifications.csv`.

## Headline

- **51 of 151 clusters classified as accreted, 34 per cent.** Belokurov & Kravtsov report
  35 per cent on their own sample, so the overall split agrees.
- We disagree with their published label on **2 clusters**: NGC 5053 and Pal 5, both
  borderline.
- **The model classified 14 clusters it had never seen, and got all 14 right.** Details
  below. This is the strongest result in the project.

## Two models

| arm | inputs | n | balanced accuracy | recall | AUC |
|---|---|---|---|---|---|
| dynamical: logistic regression | V_phi, Lz, eccentricity, log apocentre | 151 | 0.981 | 0.962 | 0.998 |
| stellar population: age-metallicity line | ages, [Fe/H] | 55 | 0.893 | 0.920 | 0.912 |

The split is deliberate. The specification asks for metallicities, ages and dynamical
information, and no single model can use all three usefully: VandenBerg's ages exist for
55 clusters while orbits reach 151. Forcing ages into the dynamical arm would discard 96
clusters. The stellar arm is also the only result here that uses no orbital information,
which matters for the circularity problem below.

## The held-out test

Belokurov & Kravtsov label 165 clusters. Harris, last updated in 2010, has 157. The 14
clusters in the first list and not the second are all recent discoveries: Gran 1 to 5,
VVV-CL001, VVV-CL160, FSR 1716, FSR 1758, Patchick 126, Laevens 3, Crater,
Sagittarius II and ESO 452-SC11. They never enter the pipeline, so they were never
trained on in any fold.

Trained on 151 and asked to classify those 14 cold, the dynamical arm got **14 out of 14
correct**, including one cluster outside the training range in V_phi and one outside it
in eccentricity.

That is a genuine generalisation result rather than another cross-validation number. It
demonstrates the pipeline reproduces the published boundary on clusters it has never
seen, at the edges of the feature distribution.

Two caveats belong next to it:

- It confirms fidelity to their *method*, not to the truth. Their labels come from the
  same orbital quantities the model is given, so a new cluster inherits whatever error
  rate their classification has. They flag 10 of 165 as possibly misclassified, and our
  model reproduces all 10 of those labels exactly.
- **The confidence is not trustworthy on poor data.** These 14 have a median fractional
  apocentre error of 0.145 against 0.031 in training, about five times worse, yet the
  model's mean confidence was essentially unchanged, 0.406 against 0.423. Since apocentre
  alone carries 0.995 AUC, a newly discovered obscured cluster with a bad distance will
  get a confidently wrong answer. The model has no way to say it does not trust an input.

## Which clusters stand out in the age-metallicity relation

- Fitting the trend and taking residuals at fixed metallicity, the five clusters youngest
  for their metallicity are **Pal 12, NGC 362, NGC 1261, NGC 4590 and NGC 1851**.
- All five are independently classified accreted by Belokurov & Kravtsov. Five of five,
  from age and metallicity alone, with no orbital information involved.
- The split only appears at fixed metallicity. A plain age cut finds nothing, because age
  and [Fe/H] are correlated across the sample.
- Clusters below the trend sit at a median 10.4 kpc, those above at 4.6 kpc.

## Rotation

- In-situ clusters rotate at a median **V_phi = +94 km/s**, accreted clusters at
  **+8 km/s**, so the accreted population is essentially not rotating with the disc.
- The disc spin-up is visible: in-situ clusters above [Fe/H] of -1.3 rotate at
  **+131 km/s** against **+75 km/s** for the metal-poor ones.
- Accreted clusters are on far more radial orbits, median eccentricity **0.77** against
  **0.52**, and reach much further out, median apocentre **28.6 kpc** against **4.0 kpc**.
- **45 of 151 clusters counter-rotate.** The least rotating are NGC 362, NGC 6712,
  NGC 6681, NGC 6981 and Pal 2, all on high-eccentricity orbits.
- This reproduces the published rotation result from an independent derivation, which is
  a check on our orbital quantities as much as a finding.

## What the dynamical arm is really measuring

Feature ablation, AUC averaged over 10 fold seeds:

| feature set | AUC | n |
|---|---|---|
| orbital features (the dynamical arm) | 0.999 | 151 |
| log apocentre alone | 0.995 | 151 |
| with [Fe/H] added back | 0.998 | 149 |
| [Fe/H] alone | 0.769 | 149 |
| [Fe/H] + age residual, no orbits | 0.928 | 55 |
| age residual alone | 0.884 | 55 |

- **A single variable, log apocentre, carries almost the entire result.** Metallicity adds
  nothing measurable and was removed from the arm, which incidentally gained 2 clusters
  and improved balanced accuracy from 0.966 to 0.981.
- **The near-perfect score is close to circular.** The reference labels are defined from
  orbital energy and angular momentum, and apocentre is a direct stand-in for energy. The
  model is re-deriving the boundary that produced the labels. Quoting 0.998 unqualified
  would badly overstate what has been shown.
- This is why the stellar arm is kept. At 0.912 AUC using no orbital information, it is
  the only evidence that the two populations differ in stellar content rather than merely
  in orbit. The age residual alone at 0.884 makes the same point.

## The contested clusters

Belokurov & Kravtsov flag 10 clusters in their comments as possibly misclassified by
their own method.

- **The dynamical arm agrees with their label on all 10.** It is structurally incapable of
  adjudicating these, because it reproduces the orbital boundary that assigned them.
- **The stellar arm disagrees on 2 of the 5 it has ages for**, and both times it lands on
  the side of the chemistry they themselves cite against their own label:

| cluster | their label | stellar arm | their note |
|---|---|---|---|
| NGC 5286 | accreted | in-situ, margin -0.36 | [Al/Fe] and [Mg/Fe] consistent with in-situ |
| NGC 7078 | in-situ | accreted, margin +0.25 | enhanced [Eu/Fe], may be accreted |

- The other five flagged clusters have no VandenBerg age, so the stellar arm cannot see
  them. The 55-cluster ceiling bites hardest exactly where the interesting cases are.
- The margins are small and this is sensitive to which stellar model is used, so it is
  suggestive rather than decisive.

## What this says about the Milky Way

- About a third of the surviving globular cluster population was accreted, two thirds
  formed in place.
- The in-situ population is centrally concentrated, median apocentre 4.0 kpc, flattened
  towards the disc plane and rotating. The accreted population extends to a median
  apocentre of 28.6 kpc, is close to spherical, and barely rotates.
- Rotation switching on above [Fe/H] of about -1.3 dates the disc's formation: clusters
  more metal-rich than that formed after the disc assembled and inherited its rotation,
  those below did not.
- Accreted clusters being younger at fixed metallicity is consistent with formation in
  lower-mass dwarf galaxies, which enrich more slowly and so reach a given metallicity
  later.

## Limits and what would come next

- The stellar arm is capped at 55 clusters by VandenBerg's age coverage, and it carries
  the only non-circular result here. Krause ages were tested and rejected: calibrating
  them onto the VandenBerg scale leaves 0.66 Gyr of residual scatter for 9 extra clusters.
- Within the Milky Way there are no unlabelled clusters left to predict on, so this is
  method validation and adjudication of contested cases rather than discovery.
- Propagating the apocentre uncertainty into the prediction would fix the confidence
  problem the held-out test exposed, and is the single most useful improvement available.
- Chemical abundances would give a third axis, though [Al/Fe] is what Belokurov & Kravtsov
  calibrated their boundary with, so it is less independent than it looks.
