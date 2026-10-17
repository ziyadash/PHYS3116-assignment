# Classification method

The chosen method is a logistic regression on a designed feature vector, implemented in
`src/method_design.py`. A two-parameter rule-based classifier was built and scored first
and is kept in `src/rule_baseline.py`, because it is the comparison that justifies
choosing the heavier model.

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
| with age, [Fe/H], position and a label | **55** (25 accreted, 30 in-situ) |

Fourteen labelled clusters do not match anything in Harris: Gran 1 to 5, VVV-CL001,
VVV-CL160, Patchick 126, FSR 1716, FSR 1758, Laevens 3, Crater, Sagittarius II,
ESO 452-SC11 and BH 140. All are discoveries postdating the 2010 Harris edition, so this
is a limitation of the catalogue rather than a failure of the name matching.

The working sample of 55 clusters is the binding constraint on every result below. It is
set by the age coverage in VandenBerg et al. (2013), not by the labels.

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

## Feature vector

| feature | reasoning | kept |
|---|---|---|
| `FeH` | accreted clusters form in lower-mass progenitors, which are less chemically enriched, so they are metal-poorer on average | yes |
| `delta_age` | age residual about the age-metallicity trend. Age and [Fe/H] are strongly correlated across the sample, so raw age largely restates [Fe/H]. The residual isolates whether a cluster is old or young for its metallicity, which is the part that separates the populations | yes |
| `log_R_gc` | in-situ clusters concentrate within roughly 10 kpc. Taken as a log because R_gc spans two decades | yes |
| `flattening` | \|Z\|/R_gc, which is dimensionless and independent of distance. It is near 0 for a disc-aligned orbit and 0.5 or above for an isotropic halo orbit | no |
| `HBtype` | the second-parameter effect reddens the horizontal branch with younger age at fixed [Fe/H], making it an indirect age proxy | no |

The last two were argued for on physical grounds and both failed empirically. `HBtype`
scores 0.463 AUC on its own, which is below chance, and `flattening` changes the full
model by -0.001 AUC. Both are still computed by `build_feature_matrix` so that the
ablation can keep demonstrating this. 

Mean standardised coefficients are `log_R_gc` +1.91, `delta_age` -1.20 and `FeH` -0.37.
The signs are physically consistent: greater distance and younger-than-trend both push a
cluster towards the accreted class.

## Results

| | balanced accuracy | recall | AUC |
|---|---|---|---|
| majority baseline | 0.500 | 0.000 | n/a |
| **logistic regression** | 0.870 | 0.840 | **0.977** |
| rule-based baseline | 0.893 | 0.920 | 0.912 |

Confusion counts, treating accreted as positive: the logistic model gives 21 true
positives, 3 false positives, 4 false negatives and 27 true negatives, and the rule
gives 23, 4, 2 and 26 respectively.

The fitted rule-based boundary is `Age = -1.47 * [Fe/H] + 9.67 Gyr`, classifying clusters
below the line as accreted.

Across 20 fold seeds the two methods are indistinguishable on balanced accuracy, at
0.884 (sd 0.016) for the logistic model against 0.884 (sd 0.019) for the rule, with the
rule ahead on 10 of 20 seeds. The rule is the better of the two on recall, at 0.890
against 0.850, winning on 13 of 20 seeds and losing on 2. On AUC the logistic model wins on 20 of 20, at
0.971 (sd 0.004) against 0.928 (sd 0.011).

Those three metrics disagree, and that disagreement is the result rather than a problem
with it. The case for the logistic model rests on AUC: it orders the clusters better,
but converts that ordering into worse decisions at the default 0.5 threshold. Tuning the
threshold is the obvious next step and should recover the decision quality without
changing the model. Reported on balanced accuracy alone, the two methods would look
equivalent and the simpler rule would be the reasonable choice.

## What the distance feature is really measuring

Feature ablation, reporting AUC averaged over 10 fold seeds:

| feature set | AUC |
|---|---|
| default: FeH, delta_age, log_R_gc | 0.971 |
| all five | 0.970 |
| **log_R_gc alone** | **0.974** |
| stellar population only: FeH, delta_age | 0.929 |
| delta_age alone | 0.884 |

Galactocentric distance on its own outperforms the entire designed feature vector. This
is not a result in favour of the model.

The correlation between `log R_gc` and the label is 0.794, and the reason is structural
rather than astrophysical. The reference labels are defined by orbital energy, and
orbital energy is largely set by how far from the Galactic centre a cluster sits. A
classifier built on `R_gc` is therefore recovering a good part of the definition of the
labels rather than discovering anything independent. Reporting 0.977 without that
qualification would overstate the result.

The defensible figure is the stellar population variant, at 0.929 AUC using no orbital
information at all. That is also the figure that answers the assignment specification,
which asks for accreted clusters to be identified from stellar populations and dynamics
rather than from a reconstruction of someone else's orbital cut.

## Next steps

1. Tune the decision threshold. The gap between AUC and balanced accuracy indicates
   performance available without any change to the features or the model.
2. Decide how the ablation is presented. The recommendation is to lead with the stellar
   population result and treat the jump from `R_gc` as a finding about what the reference
   labels encode.
3. The sample is capped at 55 by VandenBerg's age coverage. Harris Part II would supply
   [Fe/H] for all 157 clusters but no ages, so it does not lift the constraint. Whether a
   second age catalogue is worth pursuing is an open question.
4. Nothing here addresses the rotation question in the specification, which requires the
   azimuthal velocity V_phi and therefore Gaia proper motions. The options are to add a
   kinematic dataset or to scope the question out and justify that decision.
