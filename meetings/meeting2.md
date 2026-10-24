# Meeting Minutes 2
## Agenda
| Date | 7 October 2026 |
|--|--|
| Attendees | Ziyad |
| Purpose | Build the analysis pipeline: prepare the data, run the exploratory analysis, and propose classifiers to compare. |
| Next meeting | 17 October 2026 |
| Author | Ziyad |

## Progress check
- Ziyad: Cleared the whole data preparation and exploratory block from meeting 1, then went further than planned and built two working classifiers. The reference paper turned out to publish its own classifications, so we now have labels to score against. Two of the unit descriptions we wrote up last time were wrong, which was a useful reminder to check things against the numbers rather than trusting the write-up.

## Data preparation
Standardised cluster names across Harris, Krause and VandenBerg by reducing everything to
a `PREFIX+NUMBER` key. All 61 Krause and all 55 VandenBerg clusters matched onto the 157
Harris rows. One judgement call: VandenBerg writes Arp 2 as `Arp21`, which is a footnote
marker rather than a different cluster.

Two corrections to the meeting 1 data dictionary, both confirmed against the data and
fixed in those notes:

- `Mstar` is linear, in units of 10^5 solar masses. It is not a log. Reading it as a log
  would give NGC 288 a mass of 2.9 solar masses.
- `log_sigma_0` is a central surface mass density, not a velocity dispersion. It
  correlates at 0.97 with Harris central density and only 0.81 with the measured
  dispersion.

Also found that Krause and VandenBerg ages sit on different scales, differing by up to
3.4 Gyr for the same cluster, so they must not be pooled. Harris Part II is missing from
the repo, which is why metallicity is limited to 55 clusters.

## Reference labels
The paper from meeting 1 is Belokurov & Kravtsov (2024), MNRAS 528, 3198. Appendix D
publishes in-situ and accreted classifications for 165 clusters. Not available from
VizieR, so it was transcribed by hand from the PDF and verified against the PDF text
layer, which caught one transcription error.

Important limitation: their classification uses orbital energy and angular momentum,
neither of which is in our catalogues.

## Exploratory analysis
The age-metallicity relation does split, but only at fixed metallicity. A plain age cut
shows nothing because age and metallicity are correlated across the sample. Against the
fitted trend, clusters below sit at a median 10.4 kpc and those above at 4.6 kpc.

The five clusters youngest for their metallicity are Pal 12, NGC 362, NGC 1261, NGC 4590
and NGC 1851. All five are independently classified accreted in the reference table,
which is a good sign the signal is real.

Structural properties went nowhere. Horizontal branch type scores below chance on its
own and the compactness parameters add nothing.

## Method design
Built two candidates and scored them on identical folds:

- A rule-based dividing line in the age-metallicity plane, two parameters.
- Logistic regression on a designed feature vector.

Chose the logistic regression to develop further. It is comparable on balanced accuracy
but clearly better at ranking.

The important finding is a problem rather than a result. Galactocentric distance alone
outperforms the whole feature vector, because distance stands in for the orbital energy
that defines the reference labels. Any score leaning on it is close to circular. The
number worth reporting is the one using stellar populations only.

#### Action Items
**Analysis**
- [ ] Tune the decision threshold, since ranking is better than the decisions suggest.

**Data**
- [ ] Acquire orbital data so the rotation clause can be answered and the circularity
      problem removed.

**Next meeting**
- [ ] Produce an actual classification result rather than only a scoreboard.
