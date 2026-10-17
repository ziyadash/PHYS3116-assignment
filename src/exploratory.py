"""
Exploratory analysis.

Descriptive only - this module looks for structure a classifier could exploit, it does
not classify. The reference labels exist but are deliberately not used here, so that the
structure found is independent of them.
"""

import matplotlib.pyplot as plt
import numpy as np

from config import save_figure

# ---- 2.1 The age-metallicity relation ----
# Expected signature: accreted GCs are younger and less metal rich

def plot_age_metallicity(merged):
    """Plot Age against [Fe/H] and test whether it splits into two branches."""
    print('\n--- 2.1 Age-metallicity relation ---')
    sample = merged[['ID', 'FeH_vdb', 'Age_vdb', 'Age_err', 'R_gc']].dropna(
        subset=['FeH_vdb', 'Age_vdb']
    )
    print(f'    {len(sample)} clusters with both an age and a metallicity (VandenBerg)')

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].errorbar(sample['FeH_vdb'], sample['Age_vdb'], yerr=sample['Age_err'],
                     fmt='o', markersize=5, elinewidth=0.8, capsize=2, alpha=0.8)
    axes[0].set_title('Age-metallicity relation (VandenBerg et al. 2013)')

    # Colouring by R_gc tests the split independently: accreted clusters sit further
    # out, so a young branch that is also the distant branch is unlikely to be scatter.
    points = axes[1].scatter(sample['FeH_vdb'], sample['Age_vdb'], c=sample['R_gc'],
                             cmap='viridis', s=45,
                             norm=plt.matplotlib.colors.LogNorm())
    figure.colorbar(points, ax=axes[1], label='$R_{gc}$ (kpc)')
    axes[1].set_title('Coloured by galactocentric distance')

    for axis in axes:
        axis.set_xlabel('[Fe/H] (dex)')
        axis.set_ylabel('Age (Gyr)')
        axis.grid(alpha=0.3)
    save_figure(figure, 'age_metallicity.png')

    # The comparison must be made at fixed metallicity. 
    slope, intercept = np.polyfit(sample['FeH_vdb'], sample['Age_vdb'], 1)
    residual = sample['Age_vdb'] - (slope * sample['FeH_vdb'] + intercept)

    print(f'    trend: Age = {slope:+.2f} * [Fe/H] + {intercept:.2f} Gyr')
    for label, subset in [('below the trend (younger)', sample[residual < 0]),
                          ('above the trend (older)  ', sample[residual >= 0])]:
        print(f'    {label}: n = {len(subset):2d},'
              f' median R_gc = {subset["R_gc"].median():5.1f} kpc')

    # Outliers.. spec mentions it
    print('    most discrepant (youngest for their metallicity):')
    for _, row in sample.assign(residual=residual).nsmallest(5, 'residual').iterrows():
        print(f'      {row["ID"]:<12s} [Fe/H] = {row["FeH_vdb"]:+.2f},'
              f' Age = {row["Age_vdb"]:5.2f} Gyr,'
              f' {row["residual"]:+.2f} Gyr from trend,'
              f' R_gc = {row["R_gc"]:.1f} kpc')


# ---- 2.2 Spatial and kinematic distributions ----
# Expected signature: in-situ clusters inside ~10 kpc in a flattened, disc-aligned
# configuration (small |Z|); accreted clusters spread over a wide range of distances in
# a roughly spherical halo.

def plot_spatial_kinematic(merged):
    """Plot the distance and height distributions."""
    print('\n--- 2.2 Spatial and kinematic distributions ---')
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    radii = merged['R_gc'].dropna()
    axes[0].hist(radii, bins=np.logspace(np.log10(0.5), np.log10(radii.max()), 30))
    axes[0].axvline(10, color='red', linestyle='--', label='10 kpc')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('$R_{gc}$ (kpc)')
    axes[0].set_title('Galactocentric distance')
    axes[0].legend()

    axes[1].hist(merged['Z'].dropna().abs(), bins=30)
    axes[1].set_xlabel('|Z| (kpc)')
    axes[1].set_title('Height above the Galactic plane')

    # A disc-aligned inner group hugs the bottom-left; a spherical halo follows |Z| ~ R.
    axes[2].scatter(merged['R_gc'], merged['Z'].abs(), s=20, alpha=0.7)
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].set_xlabel('$R_{gc}$ (kpc)')
    axes[2].set_ylabel('|Z| (kpc)')
    axes[2].set_title('Flattening of the distribution')
    axes[2].grid(alpha=0.3)

    axes[0].set_ylabel('number of clusters')
    axes[1].set_ylabel('number of clusters')
    save_figure(figure, 'spatial_distribution.png')

    for label, subset in [('inside 10 kpc', merged[merged['R_gc'] < 10]),
                          ('beyond 10 kpc', merged[merged['R_gc'] >= 10])]:
        print(f'    {label}: {len(subset):3d} clusters,'
              f' median |Z| = {subset["Z"].abs().median():5.2f} kpc')

    # v_r is line-of-sight, so it mixes true space motion with projection geometry. The
    # rotation diagnostic in the literature is V_phi, which needs proper motions.
    print(f'    v_r available for {merged["v_r"].notna().sum()} / {len(merged)}'
          ' clusters, but it is line-of-sight only;'
          ' V_phi requires Gaia proper motions.')


# ---- 2.3 Structural properties ----
# Structural parameters respond to the tidal field a cluster has lived in, so accreted
# clusters may be systematically more diffuse at fixed mass. HBtype is the other lever:
# at fixed [Fe/H] the horizontal branch reddens with younger age (the 'second parameter'
# effect), making it an indirect age proxy.
#
# Both leads came to nothing - see the ablation in method_design - so this section is
# kept as the negative result rather than as input to the classifier.

def plot_structural_properties(merged):
    """Compare structural parameters that might separate the two populations."""
    print('\n--- 2.3 Structural properties ---')
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Equal mass but larger radius means more weakly bound - the C5 < 1 regime.
    structural = merged[['Mstar_krause', 'rh_krause', 'C5_krause']].dropna()
    points = axes[0].scatter(structural['Mstar_krause'], structural['rh_krause'],
                             c=structural['C5_krause'], cmap='plasma', s=40)
    figure.colorbar(points, ax=axes[0], label='$C_5$')
    axes[0].set_xscale('log')
    axes[0].set_xlabel(r'$M_\star$ ($10^5\,M_\odot$)')
    axes[0].set_ylabel('$r_h$ (pc)')
    axes[0].set_title(f'Mass-radius plane (n = {len(structural)})')

    # Distant clusters feel a weaker tidal field, so distance confounds any structural
    # difference between the two populations.
    concentration = merged[['c', 'R_gc']].dropna()
    axes[1].scatter(concentration['R_gc'], concentration['c'], s=20, alpha=0.7)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('$R_{gc}$ (kpc)')
    axes[1].set_ylabel('King concentration $c$')
    axes[1].set_title(f'Concentration vs distance (n = {len(concentration)})')

    branch = merged[['HBtype', 'FeH_vdb', 'Age_vdb']].dropna()
    points = axes[2].scatter(branch['FeH_vdb'], branch['HBtype'], c=branch['Age_vdb'],
                             cmap='coolwarm_r', s=45)
    figure.colorbar(points, ax=axes[2], label='Age (Gyr)')
    axes[2].set_xlabel('[Fe/H] (dex)')
    axes[2].set_ylabel('HB type (B-R)/(B+V+R)')
    axes[2].set_title(f'Second-parameter plane (n = {len(branch)})')

    for axis in axes:
        axis.grid(alpha=0.3)
    save_figure(figure, 'structural_properties.png')

    # Harris sets c = 2.5 as a core-collapse flag rather than a measurement, so these
    # clusters form a spurious horizontal line and must be excluded from any fit.
    print(f'    NOTE: c = 2.5 is a core-collapse flag, not a measurement -'
          f' {int((merged["c"] == 2.5).sum())} clusters affected.')


def exploratory_analysis(merged):
    """Produce the three exploratory figures."""
    print('\n' + '=' * 76)
    print('2. EXPLORATORY ANALYSIS')
    print('=' * 76)

    plot_age_metallicity(merged)
    plot_spatial_kinematic(merged)
    plot_structural_properties(merged)
