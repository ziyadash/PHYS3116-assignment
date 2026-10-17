"""
Sanity checks for the pipeline.

Not a unit-test suite in the proper sense, more a set of tripwires for the things that
would otherwise fail silently and poison everything downstream. Sign conventions and
coordinate frames are the obvious candidates: a flipped V_phi would still produce
perfectly reasonable-looking plots and completely wrong conclusions.

Run with:  python src/checks.py
"""

import io
import contextlib

import numpy as np

from data_prep import canonical_key, load_orbits, prepare_data
from method_design import (
    DEFAULT_FEATURES, FEATURE_SETS, ORBITAL_ONLY, ORBITAL_REQUIRED,
    REQUIRED_COLUMNS, classification_scores, evaluate, roc_auc, stratified_folds,
)

PASS, FAIL = '  ok  ', ' FAIL '
failures = []


def check(name, condition, detail=''):
    print(f'[{PASS if condition else FAIL}] {name}{"  " + detail if detail else ""}')
    if not condition:
        failures.append(name)


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        merged = prepare_data()
        orbits = load_orbits()

    print('--- name matching ---')
    # The three catalogues plus Baumgardt all spell these differently.
    check('NGC 104 forms agree',
          len({canonical_key(x) for x in ['NGC 104', 'NGC104', 'NGC_104']}) == 1)
    check('Palomar/Pal alias', canonical_key('Palomar12') == canonical_key('Pal 12'))
    check('Ruprecht/Rup alias', canonical_key('Ruprecht106') == canonical_key('Rup 106'))
    check('Arp21 override maps to Arp 2', canonical_key('Arp21') == canonical_key('Arp 2'))

    print('\n--- merge integrity ---')
    check('one row per Harris cluster', len(merged) == 157, f'got {len(merged)}')
    check('cluster IDs unique', merged['ID'].is_unique)
    check('Krause fully matched', int(merged['Mstar_krause'].notna().sum()) == 61)
    check('VandenBerg fully matched', int(merged['Age_vdb'].notna().sum()) == 55)

    print('\n--- coordinate frame ---')
    # If X, Y, Z were sun-centred rather than galactocentric this would blow up, and
    # every orbital quantity derived from them would be quietly wrong.
    radius = np.sqrt(orbits['X_gc'] ** 2 + orbits['Y_gc'] ** 2 + orbits['Z_gc'] ** 2)
    worst = float((radius - orbits['R_GC_bg']).abs().max())
    check('X,Y,Z are galactocentric', worst < 0.05, f'max error {worst:.3f} kpc')

    print('\n--- sign conventions ---')
    # The one that matters most. Belokurov & Kravtsov report in-situ clusters rotating
    # with the disc and accreted ones near zero. If our V_phi sign were flipped this
    # comparison would invert, so it doubles as an independent check on the whole
    # orbital derivation.
    labelled = merged.dropna(subset=['accreted', 'V_phi'])
    in_situ = labelled[labelled['accreted'] == 0]['V_phi'].median()
    accreted = labelled[labelled['accreted'] == 1]['V_phi'].median()
    check('in-situ rotate faster than accreted', in_situ > accreted,
          f'{in_situ:+.1f} vs {accreted:+.1f} km/s')
    check('in-situ rotation is prograde and substantial', in_situ > 50,
          f'{in_situ:+.1f} km/s')
    check('accreted rotation is near zero', abs(accreted) < 40,
          f'{accreted:+.1f} km/s')

    # Accreted clusters arrive on whatever orbit their parent had, so they should sit on
    # more radial orbits than the disc-born population.
    ecc_in = labelled[labelled['accreted'] == 0]['ecc'].median()
    ecc_acc = labelled[labelled['accreted'] == 1]['ecc'].median()
    check('accreted orbits more eccentric', ecc_acc > ecc_in,
          f'{ecc_acc:.2f} vs {ecc_in:.2f}')

    print('\n--- evaluation machinery ---')
    sample = merged.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    y = sample['accreted'].to_numpy().astype(int)

    # Every row must be tested exactly once across the folds, or the out-of-fold
    # predictions are not what they claim to be.
    seen = np.zeros(len(y), dtype=int)
    for _, test in stratified_folds(y):
        seen[test] += 1
    check('every cluster held out exactly once', bool((seen == 1).all()))

    # Folds must not overlap between train and test.
    disjoint = all(len(set(tr) & set(te)) == 0 for tr, te in stratified_folds(y))
    check('train and test folds disjoint', disjoint)

    # A perfect ranking must score 1.0 and a reversed one 0.0, otherwise the AUC
    # implementation has its comparison backwards.
    check('roc_auc on perfect ranking', abs(roc_auc(y, y.astype(float)) - 1.0) < 1e-9)
    check('roc_auc on reversed ranking', abs(roc_auc(y, -y.astype(float))) < 1e-9)
    check('roc_auc on constant scores is 0.5',
          abs(roc_auc(y, np.zeros(len(y))) - 0.5) < 1e-9)

    scores = classification_scores(y, y)
    check('scores on a perfect classifier', scores['balanced_accuracy'] == 1.0
          and scores['recall'] == 1.0 and scores['fp'] == 0 and scores['fn'] == 0)

    print('\n--- models beat chance ---')
    stellar, _, _ = evaluate(sample, FEATURE_SETS['stellar population only'])
    check('stellar-population model beats chance', stellar['roc_auc'] > 0.8,
          f'AUC {stellar["roc_auc"]:.3f}')

    orbital_sample = merged.dropna(subset=ORBITAL_REQUIRED).reset_index(drop=True)
    orbital, _, _ = evaluate(orbital_sample, ORBITAL_ONLY)
    # This one SHOULD be near perfect: the reference labels are a function of orbital
    # quantities, so anything much below this would mean we had broken the derivation.
    check('orbital model reproduces the reference labels', orbital['roc_auc'] > 0.95,
          f'AUC {orbital["roc_auc"]:.3f}')

    print('\n--- no leakage ---')
    # The age-metallicity trend is fitted per fold. If it were fitted globally the
    # held-out rows would influence their own features, so shuffling the labels should
    # collapse performance to chance. If it does not, something is leaking.
    rng = np.random.default_rng(0)
    shuffled = sample.copy()
    shuffled['accreted'] = rng.permutation(shuffled['accreted'].to_numpy())
    noise, _, _ = evaluate(shuffled, DEFAULT_FEATURES)
    check('shuffled labels score near chance', 0.3 < noise['roc_auc'] < 0.7,
          f'AUC {noise["roc_auc"]:.3f}')

    print()
    if failures:
        print(f'{len(failures)} check(s) FAILED: {", ".join(failures)}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
