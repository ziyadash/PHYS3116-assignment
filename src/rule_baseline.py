"""
Stellar population arm: a dividing line in the age-metallicity plane.

One of the project's two models. This is the arm that uses ages, which the assignment
specification names as an input and which the dynamical model cannot use, because
VandenBerg's ages cover only 55 clusters against the 149 the orbital features reach.

Keeping both arms is deliberate rather than historical. Between them they use every
input the spec names: metallicities and ages here, metallicities and dynamics in
method_design. This arm is also the only part of the project whose result does not
depend on orbital quantities, which matters given those quantities are what the
reference labels were defined from.

The rule comes directly from Belokurov & Kravtsov's result that the two populations
trace separate age-metallicity sequences, with accreted clusters younger at fixed
[Fe/H]:

    accreted  <=>  Age < slope * [Fe/H] + intercept

Both parameters come from an exhaustive grid search maximising balanced accuracy. It is
brute force, and that is the point: the grid is small enough to search outright, so
there is no arguing about whether an optimiser converged. Scoring runs on the same folds
and the same metrics as the logistic model, so the comparison is honest.
"""

import matplotlib.pyplot as plt
import numpy as np

from config import N_FOLDS, RANDOM_SEED, save_figure
from method_design import classification_scores, stratified_folds

# The rule is defined in the age-metallicity plane, so unlike the main model it is stuck
# with VandenBerg's 55 clusters. That gap is itself part of the comparison.
RULE_COLUMNS = ['accreted', 'FeH_vdb', 'Age_vdb']

# Search grid. The slope range covers every plausible orientation of the boundary and
# the intercept range spans the full age axis of the sample.
AGE_LINE_SLOPES = np.linspace(-3.0, 1.0, 81)
AGE_LINE_INTERCEPTS = np.linspace(7.0, 14.0, 141)


def fit_age_metallicity_line(feh, age, accreted):
    """Grid-search the dividing line that maximises balanced accuracy."""
    best = (-np.inf, 0.0, 0.0)
    for slope in AGE_LINE_SLOPES:
        # One row per candidate intercept, so every intercept is scored in a single
        # vectorised comparison against the whole sample.
        boundary = np.add.outer(AGE_LINE_INTERCEPTS, slope * feh)
        predictions = (age[None, :] < boundary).astype(int)
        for row, intercept in enumerate(AGE_LINE_INTERCEPTS):
            score = classification_scores(accreted, predictions[row])
            if score['balanced_accuracy'] > best[0]:
                best = (score['balanced_accuracy'], slope, intercept)
    return best[1], best[2]


def apply_age_metallicity_line(feh, age, slope, intercept):
    """Signed distance below the line. Positive means accreted."""
    return (slope * feh + intercept) - age


def evaluate_rule(sample, seed=RANDOM_SEED):
    """Cross-validate the rule, refitting the boundary inside each training fold.

    Returns (scores, mean boundary, out-of-fold margins). The margin stands in for a
    probability when computing AUC: it is a signed distance, so it ranks clusters even
    though it is not calibrated.
    """
    accreted = sample['accreted'].to_numpy().astype(int)
    feh, age = sample['FeH_vdb'].to_numpy(), sample['Age_vdb'].to_numpy()

    margin = np.zeros(len(sample))
    boundaries = []
    for train, test in stratified_folds(accreted, seed=seed):
        slope, intercept = fit_age_metallicity_line(
            feh[train], age[train], accreted[train]
        )
        boundaries.append((slope, intercept))
        margin[test] = apply_age_metallicity_line(feh[test], age[test], slope, intercept)

    scores = classification_scores(accreted, (margin > 0).astype(int), margin)
    mean_boundary = (np.mean([s for s, _ in boundaries]),
                     np.mean([i for _, i in boundaries]))
    return scores, mean_boundary, margin


def plot_decision_boundary(sample, slope, intercept):
    """Show the fitted boundary against the reference labels."""
    figure, axis = plt.subplots(figsize=(7, 5.5))
    for value, colour, label in [(0, 'tab:blue', 'in-situ'), (1, 'tab:red', 'accreted')]:
        subset = sample[sample['accreted'] == value]
        axis.scatter(subset['FeH_vdb'], subset['Age_vdb'], c=colour, s=45,
                     alpha=0.85, label=f'{label} (Belokurov & Kravtsov 2024)')

    grid = np.linspace(sample['FeH_vdb'].min(), sample['FeH_vdb'].max(), 100)
    axis.plot(grid, slope * grid + intercept, 'k--',
              label=f'fitted rule: Age = {slope:+.2f}[Fe/H] + {intercept:.2f}')
    axis.set_xlabel('[Fe/H] (dex)')
    axis.set_ylabel('Age (Gyr)')
    axis.set_title('Stellar population model: boundary vs reference labels')
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    save_figure(figure, 'decision_boundary.png')


def rule_baseline(merged):
    """Fit and report the stellar population model."""
    print('\n' + '=' * 76)
    print('4. STELLAR POPULATION MODEL (age-metallicity plane)')
    print('=' * 76)

    sample = merged.dropna(subset=RULE_COLUMNS).reset_index(drop=True)
    scores, (slope, intercept), _ = evaluate_rule(sample)

    print(f'\n{N_FOLDS}-fold stratified cross-validation, n = {len(sample)}')
    print(f'  balanced {scores["balanced_accuracy"]:.3f},'
          f' recall {scores["recall"]:.3f},'
          f' AUC {scores["roc_auc"]:.3f}')
    print(f'  confusion: TP={scores["tp"]} FP={scores["fp"]}'
          f' FN={scores["fn"]} TN={scores["tn"]}')
    print(f'  mean fitted boundary: Age = {slope:+.2f} * [Fe/H] + {intercept:.2f} Gyr'
          ' (accreted below)')

    plot_decision_boundary(sample, slope, intercept)
    return sample, scores
