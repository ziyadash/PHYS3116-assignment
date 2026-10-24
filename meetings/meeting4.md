# Meeting Minutes 4
## Agenda
| Date | 24 October 2026 |
|--|--|
| Attendees | Ziyad |
| Purpose | Cut down to two models, stop throwing away two thirds of the data, and check the two models actually cover the spec. |
| Next meeting | To be confirmed |
| Author | Ziyad |

## Progress check
- Ziyad: Realised two things that should have been obvious earlier. Running three
  separate models is more than this project needs, and the main one was only ever using
  55 clusters when orbits exist for 151. Both come back to the same root cause, so fixing
  one fixed the other.

## The problem
The designed feature vector was `FeH, delta_age, V_phi, Lz, ecc, log_R_Apo`. Two of those
come from VandenBerg et al. (2013), which only covers 55 clusters, so the whole model was
pinned to 55 even though the orbital features were available for 151.

That left the project running three models at once: the designed vector on 55, an
orbital-only model on 151 that was actually generating the output file, and a
stellar-population model on 55 used as an independence check. The classification result
was coming from a model that was never presented as the method.

## What changed
Dropped `delta_age`. It was the single most restrictive feature in the vector and it only
ever contributed on a third of the sample.

Dropping it alone was not enough, because `[Fe/H]` was also coming from VandenBerg. The
fix was Harris Part II, which had been missing from the repo since the start and carries
`[Fe/H]` for 152 of the 157 clusters. Checked it against VandenBerg on the 55 shared
clusters: mean difference +0.011 dex, scatter 0.068 dex, so the two are on the same scale
and swapping sources is safe.

The model is now `V_phi, Lz, ecc, log_R_Apo` and runs on **151 clusters** rather than 55,
a 2.7 times increase. `FeH` was in it briefly and then removed, because the ablation
showed it adds nothing over the orbital features. Taking it out gained two more clusters,
the ones with no published Harris metallicity, and improved balanced accuracy from 0.966
to 0.981. There are now two models: this one, and the rule-based baseline
kept for comparison. The old orbital-only and stellar-only models are reported as ablation
rows rather than run as separate things.

## Results
| | n | balanced accuracy | recall | AUC |
|---|---|---|---|---|
| dynamical arm | 151 | 0.981 | 0.962 | 0.998 |
| stellar population arm | 55 | 0.893 | 0.920 | 0.912 |

51 of 151 classified accreted, 34 per cent, unchanged throughout. Disagreements with
Belokurov & Kravtsov are down to two, NGC 5053 and Pal 5, both borderline.

## The held-out test
The best result of the session, and it came out of asking how the model would handle
newly discovered clusters.

Belokurov & Kravtsov label 165 clusters; Harris, unchanged since 2010, has 157. The 14 in
the first and not the second are all recent discoveries, so they never enter the pipeline
and have never been trained on. Trained on 151 and asked to classify those 14 cold, the
dynamical arm got **14 out of 14 correct**, including one cluster outside the training
range in V_phi and another outside it in eccentricity.

This now runs as part of `main.py` rather than being a one-off check, so it reproduces.

Two caveats recorded alongside it. It shows fidelity to their method, not to the truth,
since their labels come from the same orbital quantities the model is given. And the
confidence is not trustworthy on poor data: these 14 have a median fractional apocentre
error of 0.145 against 0.031 in training, five times worse, yet the model reported
essentially the same confidence. Given apocentre alone carries 0.995 AUC, a newly
discovered obscured cluster with a bad distance will get a confidently wrong answer.

## Catching a spec problem
Dropping `delta_age` nearly broke a specification requirement. The spec asks for
"metallicities [Fe/H] and ages" and dynamical information to be used in identifying
accreted clusters. With `delta_age` gone, the model used metallicity and dynamics but no
ages at all, and VandenBerg et al. (2013), a catalogue the spec names explicitly, would
have been reduced to producing exploratory plots.

The fix was already in the repo. The age-metallicity rule, previously labelled a rejected
baseline, is a model that uses exactly the inputs the dynamical arm cannot. So instead of
one method plus a discarded comparison, the project now has two arms:

| arm | inputs | n |
|---|---|---|
| stellar population: dividing line in the age-metallicity plane | ages, metallicities | 55 |
| dynamical: logistic regression | orbits | 151 |

Between them they use every input the spec names, and each runs on the largest sample its
own features allow. This is a relabelling, not new code, but it changes what the project
claims to be doing.

## What this cost
The independence claim moved rather than disappeared. The old stellar-population model
scored 0.928 AUC with no orbital information. With `delta_age` out of the dynamical arm,
`FeH` alone manages 0.769 there. But the stellar arm still scores 0.912 using no orbital
quantities at all, so the non-circular evidence survives in the arm built for it.

The ablation is also why `FeH` came back out. Orbital features alone score 0.999 and
`log_R_Apo` by itself reaches 0.995, so one feature does essentially all the work. That is
the circularity problem again: the reference labels are defined from orbital energy and
apocentre stands in for it.

| feature set | AUC | n |
|---|---|---|
| orbital only (the arm) | 0.999 | 151 |
| log_R_Apo alone | 0.995 | 151 |
| with FeH added back | 0.998 | 149 |
| FeH alone | 0.769 | 149 |
| FeH + delta_age, no orbits | 0.928 | 55 |
| delta_age alone | 0.884 | 55 |

So the trade: we gained 94 clusters in the dynamical arm and a clearer two-model
structure, and the non-circular result now rests entirely on the stellar arm's 55
clusters rather than being part of the main model.

#### Action Items
**Analysis**
- [ ] Propagate the apocentre uncertainty into the prediction, so poorly measured
      clusters return a wider probability instead of false confidence. This is the main
      weakness the held-out test exposed.
- [ ] Threshold tuning is implemented but makes balanced accuracy slightly worse, 0.970
      against 0.981. Either improve it or take it out.
- [ ] A second age catalogue is the only way past 55 clusters in the stellar arm, which
      carries the project's one non-circular result.

**Admin**
- [ ] Confirm with the professor that the project continues solo.
