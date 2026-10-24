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
NAME_PREFIX_ALIASES = {'PALOMAR': 'PAL', 'RUPRECHT': 'RUP', 'TER': 'TERZAN',
                       # Baumgardt writes Djor_1, everyone else writes Djorg 1.
                       'DJOR': 'DJORG'}

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
    harris_ii = pd.read_csv(DATA_FILEPATHS['HarrisPartII'])
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

    return harris_i, harris_ii, harris_iii, krause, vandenberg


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

def merge_catalogues(harris_i, harris_ii, harris_iii, krause, vandenberg):
    """Join the four tables into one row-per-cluster dataframe."""
    harris = harris_i.merge(harris_iii, on='ID', how='outer', validate='one_to_one')
    # Part II is where [Fe/H] lives, and it covers 152 of the 157 clusters rather than
    # VandenBerg's 55. This is what lets the classifier run on the full catalogue.
    harris = harris.merge(
        harris_ii.rename(columns={'FeH': 'FeH_harris'}), on='ID', how='left',
    )

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


# Baumgardt's orbit table, from people.smp.uq.edu.au/HolgerBaumgardt/globular/, retrieved
# 17 Oct 2026. This is the Gaia-derived one, so it finally gives us proper motions and
# full space velocities instead of the line-of-sight-only view Harris offers.
#
# Whitespace-delimited with a three-line comment header, and the header reuses names like
# 'Delta_R' for different columns, so the names below are supplied by hand in the order
# the file's own numbered (1)...(31) line defines them.
ORBIT_COLUMNS = [
    'Cluster', 'RA_bg', 'DEC_bg', 'l_bg', 'b_bg', 'Rsun_bg', 'Delta_Rsun', 'R_GC_bg',
    'RV_bg', 'ERV_bg', 'mualpha', 'Del_mualpha', 'mudelta', 'Del_mudelta', 'corr',
    'X_gc', 'Delta_X', 'Y_gc', 'Delta_Y', 'Z_gc', 'Delta_Z',
    'U_gc', 'Delta_U', 'V_gc', 'Delta_V', 'W_gc', 'Delta_W',
    'R_Per', 'Delta_R_Per', 'R_Apo', 'Delta_R_Apo',
]


def load_orbits():
    """Read Baumgardt's orbit table and derive the orbital quantities we care about."""
    orbits = pd.read_csv(
        DATA_FILEPATHS['orbits'], sep=r'\s+', comment='#', names=ORBIT_COLUMNS,
    )
    orbits['key'] = orbits['Cluster'].map(canonical_key)

    # X, Y, Z here are genuinely galactocentric!
    radius_in_plane = np.hypot(orbits['X_gc'], orbits['Y_gc'])

    # Lz, the vertical angular momentum.
    orbits['Lz'] = orbits['X_gc'] * orbits['V_gc'] - orbits['Y_gc'] * orbits['U_gc']

    # V_phi, the azimuthal velocity.
    orbits['V_phi'] = orbits['Lz'] / radius_in_plane

    # Eccentricity straight off the integrated orbit.
    orbits['ecc'] = ((orbits['R_Apo'] - orbits['R_Per'])
                     / (orbits['R_Apo'] + orbits['R_Per']))

    return orbits


def attach_orbits(merged, orbits):
    """Add the derived orbital columns to the merged catalogue."""
    keep = ['key', 'V_phi', 'Lz', 'ecc', 'R_Apo', 'Delta_R_Apo', 'R_Per',
            'mualpha', 'mudelta']
    return merged.merge(orbits[keep], on='key', how='left')


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


def load_holdout_clusters(merged):
    """Labelled clusters with orbits that are absent from Harris, so never trained on.

    These are the recent discoveries: Gran 1 to 5, the VVV-CL and FSR clusters,
    Patchick 126, Laevens 3, Crater and Sagittarius II. Harris has not been updated since
    2010, so they drop out of the main pipeline and make a genuine held-out test set
    rather than another cross-validation fold.
    """
    orbits = load_orbits()
    labels = pd.read_csv(DATA_FILEPATHS['labels'], comment='#')
    labels['key'] = labels['Name'].map(canonical_key)

    holdout = orbits.merge(labels[['key', 'insitu', 'Name']], on='key', how='inner')
    holdout = holdout[~holdout['key'].isin(merged['key'])].copy()
    holdout['accreted'] = 1 - holdout['insitu']

    # The model only reads orbital columns, but build_feature_matrix touches the stellar
    # ones too, so they are present and empty. None of these clusters has an age anyway.
    for column in ['FeH_harris', 'FeH_vdb', 'Age_vdb', 'HBtype']:
        holdout[column] = np.nan
    holdout['R_gc'] = holdout['R_GC_bg']
    holdout['Z'] = holdout['Z_gc']
    return holdout


def prepare_data():
    """Load, key, merge, label and profile the catalogues."""
    print('=' * 76)
    print('1. DATA PREPARATION')
    print('=' * 76)

    harris_i, harris_ii, harris_iii, krause, vandenberg = load_catalogues()
    merged = merge_catalogues(harris_i, harris_ii, harris_iii, krause, vandenberg)
    merged = attach_orbits(merged, load_orbits())
    merged = attach_labels(merged)

    # Match rates guard the name standardisation: anything unmatched is a key that
    # needs an alias or an override above.
    print(f'\nHarris clusters:            {len(merged)}')
    print(f'Krause matched to Harris:   {krause["key"].isin(merged["key"]).sum()}'
          f' / {len(krause)}')
    print(f'VandenBerg matched:         {vandenberg["key"].isin(merged["key"]).sum()}'
          f' / {len(vandenberg)}')
    print(f'Baumgardt orbits matched:   {int(merged["V_phi"].notna().sum())} / {len(merged)}')
    print(f'Reference labels attached:  {int(merged["accreted"].notna().sum())}')

    # Completeness of the columns that gate the analysis. Harris Part II holds [Fe/H]
    # and is not in data/, so metallicity is limited to the clusters VandenBerg covers.
    print('\nCompleteness of the columns the analysis depends on:')
    for column in ['R_gc', 'Z', 'v_r', 'FeH_harris', 'FeH_vdb', 'Age_vdb', 'HBtype',
                   'V_phi', 'Lz', 'ecc', 'accreted']:
        present = int(merged[column].notna().sum())
        print(f'  {column:<10} {present:>4} / {len(merged)}'
              f'  ({100 * present / len(merged):4.1f}%)')

    return merged
