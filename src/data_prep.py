"""
Data preparation: load the four catalogues, reconcile their cluster names, merge them
into one row-per-cluster table, and attach the reference classifications.

Harris is the base table. It is the largest catalogue (157 clusters) and the only one
carrying positions and line-of-sight velocities; Krause and VandenBerg join on as extra
columns.
"""

import re

import numpy as np
import pandas as pd

from config import DATA_FILEPATHS

# Three catalogues, three opinions on how to spell the same cluster. Harris says
# 'NGC 104' and 'Pal 12', Krause says 'NGC104' and 'Palomar12', VandenBerg just gives a
# bare number and hopes you work it out. We flatten all of it to PREFIX+NUMBER, so
# everyone ends up at 'NGC104', 'PAL12', 'TERZAN8' whether they like it or not.
NAME_PREFIX_ALIASES = {'PALOMAR': 'PAL', 'RUPRECHT': 'RUP', 'TER': 'TERZAN'}

EXPLICIT_NAME_OVERRIDES = {
    # The 'Arp21' trap. There is no Arp 21, it is Arp 2 with a footnote marker glued
    # to the end. Harris has no Arp 21 either, VandenBerg's sample includes the
    # Sagittarius clusters, and Belokurov & Kravtsov's Table D1 says 'Arp 2'. Three
    # independent witnesses, case closed.
    'ARP21': 'ARP2',
    # Belokurov & Kravtsov spell three Harris clusters differently.
    '2MASSGC01': '2MSGC01',
    '2MASSGC02': '2MSGC02',
    'ESO280SC06': 'ESOSC6',
}


def canonical_key(name):
    """Reduce a cluster name to a PREFIX+NUMBER key, or None if blank."""
    if pd.isna(name):
        return None

    cleaned = re.sub(r'[^A-Za-z0-9]', '', str(name)).upper()
    match = re.fullmatch(r'([A-Z]+)(\d+)', cleaned)
    if match is None:
        # Irregular IDs (2MS-GC01, ESO-SC06) and numberless names (Eridanus, Pyxis)
        # pass through unchanged apart from the alias lookup.
        key = cleaned or None
    else:
        prefix, number = match.groups()
        key = f'{NAME_PREFIX_ALIASES.get(prefix, prefix)}{int(number)}'

    return EXPLICIT_NAME_OVERRIDES.get(key, key)


def load_catalogues():
    """Read the four CSVs and attach a canonical 'key' column to each."""
    harris_i = pd.read_csv(DATA_FILEPATHS['HarrisPartI'])
    harris_iii = pd.read_csv(DATA_FILEPATHS['HarrisPartIII'])
    krause = pd.read_csv(DATA_FILEPATHS['Krause21'])

    # VandenBerg writes 'XXXX' for a missing NGC number or missing common name.
    vandenberg = pd.read_csv(DATA_FILEPATHS['vandenBerg'], na_values=['XXXX'])
    vandenberg = vandenberg.rename(columns={'#NGC': 'NGC'})
    vandenberg['NGC'] = pd.to_numeric(vandenberg['NGC'], errors='coerce')

    harris_i['key'] = harris_i['ID'].map(canonical_key)
    krause['key'] = krause['Object'].map(canonical_key)
    # Key on the NGC number where there is one, otherwise the common name (Arp 2,
    # Pal 12, Ter 8).
    vandenberg['key'] = np.where(
        vandenberg['NGC'].notna(),
        'NGC' + vandenberg['NGC'].astype('Int64').astype(str),
        vandenberg['Name'].map(canonical_key),
    )

    return harris_i, harris_iii, krause, vandenberg


# WARNING, DO NOT POOL THESE. Krause and VandenBerg both give Age and [Fe/H], but on
# different scales, and they disagree by as much as 3.4 Gyr on the same cluster.
# Averaging them would smear out the exact structure we are hunting for. They stay in
# separate suffixed columns. VandenBerg wins the tiebreak: the spec names it, and it
# quotes per-cluster uncertainties.
#
# Columns that are not what they look like (all three confirmed against the numbers,
# see meeting 1):
#   Mstar       linear, in 1e5 Msun. Not a log, despite what we first wrote down
#   log_sigma_0 a surface mass density, log(Msun/pc^2). The name says velocity
#               dispersion. The name is lying
#   R_G         matches Harris R_gc to 0.000 kpc, i.e. it IS R_gc wearing a hat. Dropped

def merge_catalogues(harris_i, harris_iii, krause, vandenberg):
    """Join the four tables into one row-per-cluster dataframe."""
    harris = harris_i.merge(harris_iii, on='ID', how='outer', validate='one_to_one')

    krause_cols = krause[['key', 'AltName', 'Mstar', 'rh', 'C5', 'Age', 'FeH']]
    vandenberg_cols = vandenberg[
        ['key', 'Name', 'FeH', 'Age', 'Age_err', 'HBtype', 'M_V', 'v_e0', 'log_sigma_0']
    ]

    merged = harris.merge(krause_cols, on='key', how='left', suffixes=('', '_krause'))
    merged = merged.merge(
        vandenberg_cols, on='key', how='left', suffixes=('_krause', '_vdb'),
    )

    # Merge suffixes only fire on collisions; rename the rest so every column states
    # its catalogue of origin.
    return merged.rename(columns={
        'Name_krause': 'Name',          # Harris common name
        'AltName': 'AltName_krause',
        'Mstar': 'Mstar_krause',
        'rh': 'rh_krause',
        'C5': 'C5_krause',
        'r_h': 'r_h_harris_arcmin',
    })


def attach_labels(merged):
    """Join the Belokurov & Kravtsov Table D1 classifications onto the catalogue.

    Target is `accreted` (1 = accreted), inverted from the published `insitu` flag so
    that recall refers to the class of interest.
    """
    labels = pd.read_csv(DATA_FILEPATHS['labels'], comment='#')
    labels['key'] = labels['Name'].map(canonical_key)
    labels = labels.rename(columns={'Comment': 'Comment_bk24'})

    merged = merged.merge(
        labels[['key', 'insitu', 'Comment_bk24']], on='key', how='left',
    )
    merged['accreted'] = 1 - merged['insitu']
    return merged


def prepare_data():
    """Load, key, merge, label and profile the catalogues."""
    print('=' * 76)
    print('1. DATA PREPARATION')
    print('=' * 76)

    harris_i, harris_iii, krause, vandenberg = load_catalogues()
    merged = attach_labels(merge_catalogues(harris_i, harris_iii, krause, vandenberg))

    # Match rates guard the name standardisation: anything unmatched is a key that
    # needs an alias or an override above.
    print(f'\nHarris clusters:            {len(merged)}')
    print(f'Krause matched to Harris:   {krause["key"].isin(merged["key"]).sum()}'
          f' / {len(krause)}')
    print(f'VandenBerg matched:         {vandenberg["key"].isin(merged["key"]).sum()}'
          f' / {len(vandenberg)}')
    print(f'Reference labels attached:  {int(merged["accreted"].notna().sum())}')

    # Completeness of the columns that gate the analysis. Harris Part II holds [Fe/H]
    # and is not in data/, so metallicity is limited to the clusters VandenBerg covers.
    print('\nCompleteness of the columns the analysis depends on:')
    for column in ['R_gc', 'Z', 'v_r', 'FeH_vdb', 'Age_vdb', 'HBtype', 'accreted']:
        present = int(merged[column].notna().sum())
        print(f'  {column:<10} {present:>4} / {len(merged)}'
              f'  ({100 * present / len(merged):4.1f}%)')

    return merged
