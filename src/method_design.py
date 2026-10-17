"""
Method design: logistic regression on a designed feature vector, classifying each
globular cluster as accreted or in-situ.

A ceiling applies. The reference labels come from orbital energy and angular momentum,
neither of which is in these catalogues, so the model cannot reproduce them exactly.
Worse, galactocentric distance partly proxies orbital energy, so a high score is not
automatically a stellar-population result, and `report_ablation` quantifies that.

See docs/methodology.md for the comparison and the recommendation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import N_FOLDS, RANDOM_SEED, save_figure

# Columns a cluster needs before it can be classified at all.
REQUIRED_COLUMNS = ['accreted', 'FeH_vdb', 'Age_vdb', 'R_gc', 'Z', 'HBtype',
                    'V_phi', 'Lz', 'ecc', 'R_Apo']

# Five candidate features, each intended to carry a distinct physical axis:
#
#   FeH         accreted clusters come from lower-mass progenitors, so are metal-poorer
#   delta_age   age residual about the age-metallicity trend. Age and [Fe/H] correlate
#               strongly, so raw age mostly restates [Fe/H]; the residual isolates "old
#               or young *for its metallicity*", which is the separating part
#   log_R_gc    in-situ clusters concentrate inside ~10 kpc; log because R_gc spans two
#               decades
#   flattening  |Z|/R_gc. Dimensionless and distance-free: ~0 for a disc-aligned orbit,
#               ~0.5+ for an isotropic halo one
#   HBtype      the second-parameter effect makes HB morphology an age proxy at fixed
#               [Fe/H]
FEATURE_NAMES = ['FeH', 'delta_age', 'log_R_gc', 'flattening', 'HBtype',
                 'V_phi', 'Lz', 'ecc', 'log_R_Apo']

FEATURE_SETS = {
    'stellar + orbital': [0, 1, 5, 6, 7, 8],
    'stellar population only': [0, 1],
    'orbital only': [5, 6, 7, 8],
    'old default (FeH, dAge, logRgc)': [0, 1, 2],
    'log_R_gc alone': [2],
    'delta_age alone': [1],
}
DEFAULT_FEATURES = FEATURE_SETS['stellar + orbital']

# Orbital features need no ages, so they run on 149 clusters instead of 55. Worth
# scoring separately, because tripling the sample is not nothing.
ORBITAL_ONLY = FEATURE_SETS['orbital only']
ORBITAL_REQUIRED = ['accreted', 'V_phi', 'Lz', 'ecc', 'R_Apo']


def build_feature_matrix(frame, age_trend):
    """Assemble all five candidate features. `age_trend` is fitted on the train fold."""
    slope, intercept = age_trend
    feh = frame['FeH_vdb'].to_numpy()
    radius = frame['R_gc'].to_numpy()
    return np.column_stack([
        feh,
        frame['Age_vdb'].to_numpy() - (slope * feh + intercept),
        np.log10(radius),
        np.abs(frame['Z'].to_numpy()) / radius,
        frame['HBtype'].to_numpy(),
        frame['V_phi'].to_numpy(),
        frame['Lz'].to_numpy() / 1e3,
        frame['ecc'].to_numpy(),
        np.log10(frame['R_Apo'].to_numpy()),
    ])


# ---- Model ----
def fit_logistic_regression(x, y, l2_penalty=1.0, iterations=4000, learning_rate=0.2):
    """Logistic regression by gradient descent, with the intercept left unpenalised."""
    design = np.column_stack([np.ones(len(x)), x])
    weights = np.zeros(design.shape[1])
    for _ in range(iterations):
        probability = 1.0 / (1.0 + np.exp(-design @ weights))
        gradient = design.T @ (probability - y) / len(y)
        gradient[1:] += l2_penalty * weights[1:] / len(y)
        weights -= learning_rate * gradient
    return weights


def predict_logistic(weights, x):
    """Return P(accreted) for each row of x."""
    return 1.0 / (1.0 + np.exp(-np.column_stack([np.ones(len(x)), x]) @ weights))


# ---- Evaluation ----
# Stratified k-fold keeps the class balance in every fold, which matters at n = 55.
# The age-metallicity trend and the feature scaling are fitted inside the training fold
# only, otherwise the test rows leak into the features.

def stratified_folds(y, n_splits=N_FOLDS, seed=RANDOM_SEED):
    """Yield (train_index, test_index) pairs preserving the class ratio."""
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)
    for class_value in np.unique(y):
        members = np.flatnonzero(y == class_value)
        rng.shuffle(members)
        fold_of[members] = np.arange(len(members)) % n_splits
    for fold in range(n_splits):
        yield np.flatnonzero(fold_of != fold), np.flatnonzero(fold_of == fold)


def roc_auc(y_true, scores):
    """ROC AUC via the rank-sum identity (equivalent to the Mann-Whitney statistic)."""
    n_positive, n_negative = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n_positive == 0 or n_negative == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average the ranks of tied scores so ties contribute 0.5 rather than 0 or 1.
    for value in np.unique(scores):
        tied = scores == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return (ranks[y_true == 1].sum() - n_positive * (n_positive + 1) / 2) / (
        n_positive * n_negative
    )


def classification_scores(y_true, y_predicted, scores=None):
    """Balanced accuracy, recall, the confusion counts.. ROC AUC."""
    counts = {
        'tp': int(((y_predicted == 1) & (y_true == 1)).sum()),
        'tn': int(((y_predicted == 0) & (y_true == 0)).sum()),
        'fp': int(((y_predicted == 1) & (y_true == 0)).sum()),
        'fn': int(((y_predicted == 0) & (y_true == 1)).sum()),
    }
    recall = counts['tp'] / max(counts['tp'] + counts['fn'], 1)
    specificity = counts['tn'] / max(counts['tn'] + counts['fp'], 1)

    # Keeping the raw counts means anyone who misses precision can rebuild it in their
    # head: tp / (tp + fp). No information was harmed in the making of this cut.
    return counts | {
        'balanced_accuracy': 0.5 * (recall + specificity),
        'recall': recall,
        'roc_auc': roc_auc(y_true, scores) if scores is not None else None,
    }


def best_threshold(y_true, probability):
    """Pick the cut that maximises balanced accuracy. Fitted on training folds only."""
    best, best_score = 0.5, -np.inf
    for cut in np.linspace(0.05, 0.95, 91):
        score = classification_scores(y_true, (probability > cut).astype(int))
        if score['balanced_accuracy'] > best_score:
            best, best_score = cut, score['balanced_accuracy']
    return best


def evaluate(sample, columns, seed=RANDOM_SEED, tune_threshold=False, required=None):
    """Cross-validate the model on one feature subset.

    Returns (scores, mean coefficients, out-of-fold P(accreted)).
    """
    accreted = sample['accreted'].to_numpy().astype(int)
    feh, age = sample['FeH_vdb'].to_numpy(), sample['Age_vdb'].to_numpy()

    probability = np.zeros(len(sample))
    cut = np.full(len(sample), 0.5)
    weights_per_fold = []
    for train, test in stratified_folds(accreted, seed=seed):
        # When ages are absent (the orbital-only run) the trend is unused, so a flat
        # placeholder keeps build_feature_matrix happy without touching the result.
        trend = (np.polyfit(feh[train], age[train], 1)
                 if np.isfinite(feh[train]).all() and np.isfinite(age[train]).all()
                 else (0.0, 0.0))
        x_train = build_feature_matrix(sample.iloc[train], trend)[:, columns]
        x_test = build_feature_matrix(sample.iloc[test], trend)[:, columns]
        centre, spread = x_train.mean(axis=0), x_train.std(axis=0)
        spread[spread == 0] = 1.0

        weights = fit_logistic_regression((x_train - centre) / spread, accreted[train])
        weights_per_fold.append(weights)
        probability[test] = predict_logistic(weights, (x_test - centre) / spread)

        if tune_threshold:
            # Tuned on the training fold's own predictions, never on the test rows.
            train_probability = predict_logistic(weights, (x_train - centre) / spread)
            cut[test] = best_threshold(accreted[train], train_probability)

    scores = classification_scores(accreted, (probability > cut).astype(int), probability)
    return scores, np.mean(weights_per_fold, axis=0), probability


def report_ablation(sample, seeds=range(10)):
    """Score each feature subset so the contribution of each feature is visible.

    Averaged over several fold seeds, because at n = 55 a single split is noisy.
    """
    seeds = list(seeds)
    print(f'\n--- Feature ablation (AUC, mean over {len(seeds)} fold seeds) ---')
    for label, columns in FEATURE_SETS.items():
        areas = [evaluate(sample, columns, seed=s)[0]['roc_auc'] for s in seeds]
        print(f'  {label:<30} {np.mean(areas):.3f} +/- {np.std(areas):.3f}')


# ---- Reporting ----
def plot_predictions(sample, probability):
    """Out-of-fold P(accreted) in the age-metallicity plane, against the true labels."""
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    points = axis.scatter(
        sample['FeH_vdb'], sample['Age_vdb'], c=probability,
        cmap='coolwarm', vmin=0, vmax=1, s=70,
        edgecolors=['darkred' if a else 'navy' for a in sample['accreted']],
        linewidths=1.6,
    )
    figure.colorbar(points, ax=axis, label='predicted P(accreted)')
    axis.set_xlabel('[Fe/H] (dex)')
    axis.set_ylabel('Age (Gyr)')
    axis.set_title('Out-of-fold predictions (fill) vs true label (edge:'
                   ' red = accreted, blue = in-situ)', fontsize=10)
    axis.grid(alpha=0.3)
    save_figure(figure, 'predictions.png')


# ---- 3.5 The actual answer ----
# Everything above measures how well the method works. This bit finally uses it: it
# classifies every cluster it can and writes the per-cluster verdict out to a file, so
# the project has a result rather than just a scoreboard.

def classify_all(merged, output_path=None):
    """Classify every cluster we have features for and write the results out."""
    from config import PROJECT_ROOT
    output_path = output_path or PROJECT_ROOT / 'results' / 'classifications.csv'

    stellar = merged.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    orbital = merged.dropna(subset=ORBITAL_REQUIRED).reset_index(drop=True)

    # Two models, because they answer different questions. The orbital one covers far
    # more clusters but mostly re-derives Belokurov & Kravtsov's own boundary; the
    # stellar one is the independent opinion.
    _, _, p_orbital = evaluate(orbital, ORBITAL_ONLY, tune_threshold=True)
    _, _, p_stellar = evaluate(stellar, FEATURE_SETS['stellar population only'],
                               tune_threshold=True)

    table = orbital[['ID', 'Name', 'FeH_vdb', 'Age_vdb', 'R_gc', 'V_phi', 'ecc',
                     'accreted', 'Comment_bk24']].copy()
    table['p_accreted_orbital'] = p_orbital
    table = table.merge(
        stellar[['ID']].assign(p_accreted_stellar=p_stellar), on='ID', how='left',
    )
    table['our_class'] = np.where(table['p_accreted_orbital'] > 0.5,
                                  'accreted', 'in-situ')
    table['reference_class'] = np.where(table['accreted'] == 1, 'accreted', 'in-situ')
    table['agrees'] = table['our_class'] == table['reference_class']

    output_path.parent.mkdir(exist_ok=True)
    table.sort_values('p_accreted_orbital', ascending=False).to_csv(
        output_path, index=False, float_format='%.4f',
    )

    accreted_count = int((table['our_class'] == 'accreted').sum())
    print(f'\n--- Classification of all {len(table)} clusters with orbits ---')
    print(f'  accreted : {accreted_count:3d}  ({100 * accreted_count / len(table):.0f} per cent)')
    print(f'  in-situ  : {len(table) - accreted_count:3d}')
    print(f'  disagreements with Belokurov & Kravtsov: {int((~table["agrees"]).sum())}')
    print(f'  written to {output_path.relative_to(PROJECT_ROOT)}')

    disagree = table[~table['agrees']]
    if len(disagree):
        print('  where we differ:')
        for _, r in disagree.iterrows():
            flag = ' (they flagged it too)' if pd.notna(r['Comment_bk24']) else ''
            print(f'    {r["ID"]:<12s} we say {r["our_class"]:<9s}'
                  f' they say {r["reference_class"]:<9s}'
                  f' P={r["p_accreted_orbital"]:.3f}{flag}')

    return table


def method_design(merged):
    """Fit and report the classifier."""
    print('\n' + '=' * 76)
    print('3. METHOD DESIGN')
    print('=' * 76)

    sample = merged.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    accreted = sample['accreted'].to_numpy().astype(int)

    print(f'\nLabelled clusters in Harris:   {int(merged["accreted"].notna().sum())}')
    print(f'With the full feature vector:  {len(sample)}'
          f'  ({int(accreted.sum())} accreted, {int((1 - accreted).sum())} in-situ)')

    scores, weights, probability = evaluate(sample, DEFAULT_FEATURES)
    baseline = classification_scores(
        accreted, np.full(len(sample), int(round(accreted.mean())))
    )

    print(f'\n--- {N_FOLDS}-fold stratified cross-validation ---')
    print(f'{"":<22}{"bal acc":>9}{"recall":>8}{"AUC":>7}')
    print('-' * 46)
    for label, r in [('majority baseline', baseline), ('logistic (3 feat)', scores)]:
        auc = f'{r["roc_auc"]:.3f}' if r['roc_auc'] is not None else '   -  '
        print(f'{label:<22}{r["balanced_accuracy"]:>9.3f}{r["recall"]:>8.3f}{auc:>7}')
    print(f'\nConfusion (accreted = positive):  TP={scores["tp"]} FP={scores["fp"]}'
          f' FN={scores["fn"]} TN={scores["tn"]}')

    print('\nMean standardised coefficients (positive favours accreted):')
    for name, weight in zip([FEATURE_NAMES[i] for i in DEFAULT_FEATURES], weights[1:]):
        print(f'  {name:<12} {weight:+.3f}')

    report_ablation(sample)

    # Galactocentric distance and orbital energy are closely correlated
    correlation = np.corrcoef(np.log10(sample['R_gc']), accreted)[0, 1]
    print(f'\n  corr(log R_gc, accreted) = {correlation:.3f}'
          '  <- log_R_gc proxies orbital energy, which defined the labels')

    plot_predictions(sample, probability)
    classify_all(merged)
    return sample, scores
