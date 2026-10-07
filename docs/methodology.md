# Classification method

**Decision: logistic regression on a designed feature vector.** Implemented in
`src/method_design.py`. A rule-based alternative was built and scored first; it was
rejected and its code removed, but its numbers are kept below because they justify the
choice and belong in the report.

## What we classify against

`data/BelokurovKravtsov24_tableD1.csv` — transcribed from Appendix D of Belokurov &
Kravtsov (2024), MNRAS 528, 3198. 165 clusters, `1` = in-situ, `0` = accreted. Our target
is `accreted = 1 - insitu`, so recall refers to the class of interest.

The transcription reproduces the paper's stated counts exactly: 107 in-situ / 58 accreted
over 165 rows, and 106 / 58 excluding BH 140, which the paper notes has no metallicity.
164 of 165 rows were machine-verified against the PDF text layer.

**Their labels come from orbital energy `E` and angular momentum `Lz`.** Neither is in our
catalogues and neither is computable without Gaia proper motions. This caps what any
method here can do, and causes the problem in "The circularity problem" below.

### Sample

| | n |
|---|---|
| Harris clusters | 157 |
| …with a Belokurov & Kravtsov label | 151 |
| …with age, [Fe/H], position **and** a label | **55** (25 accreted, 30 in-situ) |

The 14 labelled clusters that do not match Harris (Gran 1–5, VVV-CL001/160, Patchick 126,
FSR 1716/1758, Laevens 3, Crater, Sagittarius II, ESO 452-SC11, BH 140) are post-2010
discoveries absent from the 2010 Harris edition — not a name-matching failure.

n = 55 is the binding constraint on everything below.

## Protocol

5-fold stratified cross-validation. The age–metallicity trend and the feature scaling are
both fitted **inside the training fold**, so the age-residual feature cannot leak test
rows. Where stability is quoted, scores are averaged over 20 fold seeds, because a single
split at n = 55 is noisy.

## Feature vector

| feature | rationale | kept? |
|---|---|---|
| `FeH` | accreted clusters form in lower-mass progenitors, so are metal-poorer | yes |
| `delta_age` | age residual about the age–metallicity trend. Age and [Fe/H] correlate strongly, so raw age mostly restates [Fe/H]; the residual isolates "old or young *for its metallicity*" | yes |
| `log_R_gc` | in-situ clusters concentrate inside ~10 kpc; log because R_gc spans two decades | yes |
| `flattening` = \|Z\|/R_gc | ~0 for a disc-aligned orbit, ~0.5+ for an isotropic halo one | **no** — adds −0.001 AUC |
| `HBtype` | second-parameter effect makes HB morphology an age proxy at fixed [Fe/H] | **no** — 0.463 AUC alone, below chance |

Both rejected features were justified on physical grounds and both failed empirically.
They are still computed by `build_feature_matrix` so the ablation can keep demonstrating
that, which is a reportable result in itself.

Mean standardised coefficients: `log_R_gc` +1.91, `delta_age` −1.20, `FeH` −0.37. Signs are
physically right — larger radius and younger-than-trend both push towards accreted.

## Results

| | accuracy | balanced acc | recall | precision | AUC |
|---|---|---|---|---|---|
| majority baseline | 0.545 | 0.500 | 0.000 | 0.000 | — |
| **logistic (3 feat)** | 0.873 | 0.870 | 0.840 | 0.875 | **0.977** |
| *(rejected) age–[Fe/H] line* | 0.891 | 0.893 | 0.920 | 0.852 | 0.912 |

Confusion, accreted positive: TP 21, FP 3, FN 4, TN 27.

Over 20 fold seeds the two were indistinguishable on balanced accuracy (0.879 ± 0.014 vs
0.884 ± 0.019, the line ahead on 10 of 20 seeds) but the logistic model won on AUC on
**20 of 20** (0.970 ± 0.004 vs 0.928 ± 0.011). It ranks clusters better but wastes that at
the default 0.5 threshold — **tuning the threshold is the obvious next step.**

## The circularity problem

Feature ablation (AUC, mean over 10 fold seeds):

| feature set | AUC |
|---|---|
| default (FeH, dAge, logRgc) | 0.971 |
| all five | 0.970 |
| **log_R_gc alone** | **0.974** |
| stellar population only (FeH, dAge) | 0.929 |
| delta_age alone | 0.884 |

**A single variable — galactocentric distance — outperforms the whole feature vector.**

`corr(log R_gc, accreted) = 0.794`, and the reason is structural: the reference labels are
defined by orbital energy, and orbital energy is largely set by how far out a cluster sits.
A classifier leaning on `R_gc` substantially rediscovers the definition of the labels
rather than finding independent astrophysics. A marker could reasonably call this out, so
**we should raise it ourselves rather than quote 0.977 unqualified.**

The defensible number, with no orbital information at all, is the stellar-population-only
variant: **AUC 0.929**. That is the figure that answers the spec, which asks us to use
stellar populations and dynamics to identify accreted clusters — not to reverse-engineer
someone else's orbital cut.

## Next steps

1. Tune the decision threshold — the AUC/accuracy gap says there is free performance there.
2. Decide how to present the ablation. Recommended: lead with the stellar-population result
   and present the `R_gc` jump as a finding about what the labels encode.
3. n = 55 is set by VandenBerg's age coverage. Harris Part II would give [Fe/H] for all 157
   but no ages, so it does not lift the constraint — is a second age catalogue worth chasing?
4. Nothing here addresses the spec's rotation question, which needs V_φ and so Gaia proper
   motions. Add a kinematic dataset, or scope it out and justify that.
