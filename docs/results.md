# Results

Brief notes, not a report. Numbers are reproducible by running `python src/main.py`.
Per-cluster output is in `results/classifications.csv`.

## Headline

- **50 of 149 clusters classified as accreted, 34 per cent.** Belokurov & Kravtsov get
  35 per cent on their own sample, so the two agree on the overall split.
- We disagree with their published label on **3 clusters only**: NGC 5053, NGC 5272 and
  Pal 5. All three are borderline, with predicted probabilities between 0.36 and 0.49.

## Which clusters stand out in the age-metallicity relation

- Fitting the overall trend and taking residuals at fixed metallicity, the five clusters
  youngest for their metallicity are **Pal 12, NGC 362, NGC 1261, NGC 4590 and
  NGC 1851**.
- All five are independently classified accreted by Belokurov & Kravtsov. Five out of
  five, from age and metallicity alone, with no orbital information used.
- The split is only visible at fixed metallicity. A plain age cut finds nothing, because
  age and [Fe/H] are correlated across the whole sample.
- Clusters below the trend sit at a median 10.4 kpc, those above at 4.6 kpc.

## Rotation

- In-situ clusters rotate at a median **V_phi = +94 km/s**. Accreted clusters sit at
  **+8 km/s**, essentially not rotating with the disc at all.
- The disc spin-up is visible: in-situ clusters with [Fe/H] > -1.3 rotate at
  **+131 km/s**, against **+75 km/s** for the metal-poor in-situ ones.
- Accreted clusters are on much more radial orbits, median eccentricity **0.77** against
  **0.52**, and reach far further out, median apocentre **28.6 kpc** against **4.0 kpc**.
- **44 of 149 clusters counter-rotate.** The least-rotating are NGC 362, NGC 6712,
  NGC 6681, NGC 6981 and Pal 2, all on high-eccentricity orbits.
- This reproduces Belokurov & Kravtsov's reported rotation result from an independent
  derivation, which is a check on our orbital quantities as much as a result.

## Classifier performance

| model | n | AUC | balanced accuracy | recall |
|---|---|---|---|---|
| stellar populations only | 55 | 0.929 | 0.893 | 0.920 |
| orbital only | 149 | 0.998 | 0.972 | 0.943 |

- The orbital model is **near perfect, and that is the problem, not the achievement.**
  The reference labels are defined from orbital energy and angular momentum, so a model
  given orbital quantities is re-deriving the definition of the labels. It confirms our
  pipeline is correct and tells us nothing astrophysical.
- `log R_Apo` correlates +0.821 with the label and `log R_gc` +0.794, both acting as
  stand-ins for orbital energy. `Lz` alone correlates only -0.016, because the published
  boundary is a two-dimensional shape in the energy and angular momentum plane rather
  than a cut on either axis.
- **The one independent number is 0.929 AUC from stellar populations alone.** Age and
  metallicity carry real information about accretion that is not simply orbit restated.
- Feature ablation killed two features that were argued for on physical grounds:
  `HBtype` scores 0.463 alone, below chance, and `flattening` moves the full model by
  -0.001.
- Nested cross-validation puts the optimism from choosing features on the scoring folds
  at only **+0.005 AUC**, so the reported figures are not inflated.

## The contested clusters

- Belokurov & Kravtsov flag 10 clusters in their own comments as possibly misclassified.
  Five fall inside our 55-cluster stellar sample.
- Our stellar-population model disagrees with their published label on **3 of those 5**,
  and in every case it lands on the same side as the chemical abundance evidence they
  themselves cite against their orbital classification:

| cluster | their label | our P(accreted) | their comment |
|---|---|---|---|
| NGC 7078 | in-situ | 0.783 | enhanced [Eu/Fe], may be accreted |
| NGC 6584 | accreted | 0.468 | [Al/Fe] and [Mg/Fe] consistent with in-situ |
| NGC 5286 | accreted | 0.393 | [Al/Fe] and [Mg/Fe] consistent with in-situ |

- Three of our seven total disagreements land on the five flagged clusters, roughly a
  fivefold enrichment over chance. With n = 5 this is suggestive, not conclusive.

## What this says about the Milky Way

- About a third of the surviving globular cluster population was accreted, two thirds
  formed in place.
- The in-situ population is centrally concentrated, median apocentre 4.0 kpc, flattened
  towards the disc plane, and rotating. The accreted population is spread out to a median
  apocentre of 28.6 kpc, close to spherical, and barely rotating.
- The rotation switching on above [Fe/H] of about -1.3 dates the formation of the disc:
  clusters more metal-rich than that formed after the disc had assembled and inherited
  its rotation, those below did not.
- The accreted clusters being younger at fixed metallicity is consistent with formation
  in lower-mass dwarf galaxies, which enrich more slowly and so reach a given metallicity
  later.

## Limits and what would come next

- The age sample is capped at 55 clusters by VandenBerg's coverage, which is the binding
  constraint on every stellar-population result here.
- Every Milky Way cluster we can classify is already labelled, so this is method
  validation and adjudication of contested cases, not discovery. No unlabelled population
  exists to predict on.
- A second age catalogue would extend the independent stellar-population test beyond 55.
- Chemical abundances, specifically [Al/Fe] and [Mg/Fe], would give a third genuinely
  independent axis, and would settle the contested clusters properly rather than
  suggestively.
