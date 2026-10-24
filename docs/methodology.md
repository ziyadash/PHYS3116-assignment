# Classification method

The project uses **two models**, one per kind of input the assignment specification
names.

| arm | model | inputs | n |
|---|---|---|---|
| stellar population | dividing line in the age-metallicity plane (`src/rule_baseline.py`) | ages, metallicities | 55 |
| dynamical | logistic regression on orbital quantities (`src/method_design.py`) | orbits | 151 |

Splitting them this way is deliberate. The specification asks for metallicities, ages and
dynamical information to be used, and no single model can use all three on a useful
sample: VandenBerg's ages exist for only 55 clusters, while orbits reach 151. Forcing
ages into the dynamical model would discard 96 clusters, and dropping the stellar arm
entirely would mean never using ages at all. Running both covers every named input and
keeps each model on the largest sample its own features allow.

The stellar arm matters for a second reason. It is the only result here that does not
depend on orbital quantities, and the reference labels were themselves defined from
orbital energy and angular momentum. Everything the dynamical arm achieves is therefore
partly circular; the stellar arm is not.

## What the classifier is scored against

`data/BelokurovKravtsov24_tableD1.csv` holds the classifications from Appendix D of
Belokurov & Kravtsov (2024), MNRAS 528, 3198. It covers 165 clusters, with `1` for
in-situ and `0` for accreted. The target variable used here is `accreted = 1 - insitu`,
so that recall refers to the class of interest.

The table was transcribed by hand from the published PDF, since it is not available from
VizieR or CDS. It reproduces the paper's own stated counts exactly: 107 in-situ and 58
accreted across all 165 rows, and 106 against 58 once BH 140 is excluded, which the paper
notes has no metallicity estimate. 164 of the 165 rows were additionally verified against
the PDF text layer, which caught one transcription error in ESO 280-SC06.

Their classification is derived from orbital energy `E` and the z-component of angular
momentum `Lz`. Neither quantity appears in the three catalogues used here, and neither
can be computed without Gaia proper motions. That places a ceiling on what any method in
this project can achieve, and it is the root of the problem described under "What the
distance feature is really measuring".

### Sample

| | n |
|---|---|
| Harris clusters | 157 |
| with a Belokurov & Kravtsov label | 151 |
| with orbits and a label (dynamical arm) | **151** (53 accreted, 98 in-situ) |
| with ages, metallicity and a label (stellar arm) | **55** (25 accreted, 30 in-situ) |

Fourteen labelled clusters do not match anything in Harris: Gran 1 to 5, VVV-CL001,
VVV-CL160, Patchick 126, FSR 1716, FSR 1758, Laevens 3, Crater, Sagittarius II,
ESO 452-SC11 and BH 140. All are discoveries postdating the 2010 Harris edition, so this
is a limitation of the catalogue rather than a failure of the name matching.

The two arms are capped by different things. The dynamical arm reaches 151, limited only
by the six clusters with no orbital data at all. The stellar arm is held to 55 by the age
coverage in VandenBerg et al. (2013), which is the binding constraint on every result
that uses ages.

## Evaluation protocol

Scoring uses 5-fold stratified cross-validation, with the same folds for every method so
that results are directly comparable. Stratification matters at this sample size: an
unstratified split can easily produce a fold with a badly skewed class ratio.

The age-metallicity trend and the feature scaling are both fitted inside the training
fold rather than on the full sample. Fitting them globally would let the held-out rows
influence their own features through the trend line, which inflates the reported scores.
Where stability is quoted, figures are averaged over 20 fold seeds, because a single
split at n = 55 is noisy enough to move balanced accuracy by several points.

### Choice of metrics

Three metrics are reported: balanced accuracy, recall on the accreted class, and ROC
AUC. The confusion counts are given alongside them.

## Feature vector, dynamical arm

| feature | reasoning |
|---|---|
| `V_phi` | azimuthal velocity. In-situ clusters rotate with the disc, accreted ones do not |
| `Lz` | vertical angular momentum, one of the two axes the reference classification uses |
| `ecc` | orbital eccentricity. Accreted clusters arrive on more radial orbits |
| `log_R_Apo` | log apocentre, which stands in for orbital energy |

Three candidates were tested and dropped. `delta_age`, the age residual about the
age-metallicity trend, is the strongest stellar feature available at 0.884 AUC alone, but
it exists for only 55 clusters and was holding the whole model to that sample. It now
lives in the stellar arm, where ages are the point rather than a constraint. `HBtype`
scores 0.463 alone, below chance, and `flattening` moved the model by -0.001.

`FeH` was in the vector briefly and removed once the ablation showed it adds nothing over
the orbital features. Metallicity now appears only in the stellar arm.

## Results

| | n | balanced accuracy | recall | AUC |
|---|---|---|---|---|
| majority baseline | 151 | 0.500 | 0.000 | n/a |
| **dynamical arm** | 151 | 0.981 | 0.962 | 0.998 |
| **stellar population arm** | 55 | 0.893 | 0.920 | 0.912 |

Confusion counts, accreted as positive: the dynamical arm gives 51 true positives, 0
false positives, 2 false negatives and 98 true negatives. The stellar arm gives 23, 4, 2
and 26.

The stellar arm's fitted boundary is `Age = -1.47 * [Fe/H] + 9.67 Gyr`, classifying
clusters below the line as accreted.

Applied across the catalogue, the dynamical arm classifies **51 of 151 clusters as
accreted, 34 per cent**, against the 35 per cent reported by Belokurov & Kravtsov. The
two disagree on two clusters, NGC 5053 and Pal 5, both borderline.

The two arms are not directly comparable, because they run on different samples and
different inputs. The dynamical arm is far more accurate, but see below for why that is
less impressive than it looks.

## Held-out test

The 14 clusters Belokurov & Kravtsov label but Harris does not contain, all discovered
after the 2010 Harris edition, never enter the pipeline and so were never trained on.
Trained on 151 and asked to classify those 14 cold, the dynamical arm gets **14 of 14
correct**, including one cluster outside the training range in `V_phi` and another outside
it in `ecc`. Reproduced by `holdout_check` in `src/method_design.py`.

This measures generalisation on data the model never touched, unlike the cross-validated
figures. It demonstrates fidelity to the published method, not to the truth, and it
exposed the model's main weakness: these clusters have a median fractional apocentre error
of 0.145 against 0.031 in training, yet the model reports essentially the same confidence.

## What the dynamical arm is really measuring

Feature ablation, AUC averaged over 10 fold seeds:

| feature set | AUC | n |
|---|---|---|
| orbital features (the dynamical arm) | 0.999 | 151 |
| `log_R_Apo` alone | 0.995 | 151 |
| with `FeH` added back | 0.998 | 149 |
| `FeH` alone | 0.769 | 149 |
| FeH + delta_age, no orbital information | 0.928 | 55 |
| `delta_age` alone | 0.884 | 55 |

Two things fall out of this and both belong in the report.

**Metallicity contributes nothing to the dynamical arm, so it was removed.** Orbital
features alone score 0.999 and apocentre by itself reaches 0.995. Taking `FeH` out also
gained the two clusters with no published Harris metallicity and lifted balanced accuracy
from 0.966 to 0.981.

**The near-perfect score is close to circular.** The reference labels are defined from
orbital energy and angular momentum, and apocentre is a direct stand-in for energy. A
model handed those quantities is re-deriving the boundary that produced the labels rather
than discovering anything independent. Quoting 0.998 without that qualification would
overstate the result considerably.

This is why the stellar arm is kept. At 0.912 AUC on 55 clusters, using no orbital
information whatsoever, it is the only evidence here that the two populations differ in
their stellar content rather than merely in their orbits. The ablation row showing
`delta_age` alone at 0.884 makes the same point: age carries real information about
accretion independent of dynamics.

## Next steps

1. Propagate the apocentre uncertainty into the prediction. The held-out test showed the
   model reports the same confidence on data five times worse, which is its main weakness.
2. A second age catalogue is the only way to lift the stellar arm past 55 clusters, and
   that arm carries the one non-circular result in the project. Krause ages were tested
   and rejected: calibrating them onto the VandenBerg scale leaves 0.66 Gyr of residual
   scatter for only 9 extra clusters.
3. Chemical abundances, specifically [Al/Fe] and [Mg/Fe], would give a third axis
   independent of both arms. The catch is that [Al/Fe] is what Belokurov & Kravtsov
   calibrated their boundary with, so it is not as independent as it first appears.
4. Threshold tuning is implemented but makes balanced accuracy slightly worse, 0.970
   against 0.981 untuned, because it overfits the training folds. Worth either improving
   or dropping rather than leaving in place unused.
