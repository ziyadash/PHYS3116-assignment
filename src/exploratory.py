"""
Exploratory analysis.

Descriptive only. This module looks for structure a classifier could exploit; it does
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

    # Do this at fixed metallicity or it tells you nothing. Age and [Fe/H] already
    # march together across the sample, so slicing on raw age just rediscovers that
    # correlation and looks like a result. Residuals about the trend strip it out.
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
# Spoiler: neither lead went anywhere, as the ablation in method_design cheerfully
# confirms. Kept anyway, because "we looked and there was nothing there" is a real
# result and the report will want it.

def plot_structural_properties(merged):
    """Compare structural parameters that might separate the two populations."""
    print('\n--- 2.3 Structural properties ---')
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Equal mass but larger radius means more weakly bound, the C5 < 1 regime.
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


# ---- 2.4 Rotation ----
# Using the Baumgardt dataset, now we have data about 'proper motions', so we can 
# get V_phi and do some exploration about rotational motion.
#
# Per Belokurov & Kravtsov: in-situ clusters rotate along with the disc,
# accreted ones sit near zero because their parent dwarf arrived on whatever orbit it
# happened to arrive on. The metal-rich in-situ clusters should show the strongest
# rotation, which is the "disc spin-up" they describe.

def plot_rotation(merged):
    """Plot the rotation diagnostic and name the clusters that do not rotate."""
    print('\n--- 2.4 Rotation ---')
    sample = merged.dropna(subset=['V_phi', 'R_gc'])
    print(f'    {len(sample)} clusters with a space velocity (Baumgardt)')

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].hist(sample['V_phi'], bins=30)
    axes[0].axvline(0, color='red', linestyle='--', label='no rotation')
    axes[0].set_xlabel(r'$V_\phi$ (km/s)')
    axes[0].set_ylabel('number of clusters')
    axes[0].set_title('Rotation velocity')
    axes[0].legend()

    # The spin-up plot. Rotation should switch on above [Fe/H] of about -1.3.
    chem = merged.dropna(subset=['V_phi', 'FeH_vdb'])
    axes[1].scatter(chem['FeH_vdb'], chem['V_phi'], s=35, alpha=0.8)
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_xlabel('[Fe/H] (dex)')
    axes[1].set_ylabel(r'$V_\phi$ (km/s)')
    axes[1].set_title(f'Disc spin-up (n = {len(chem)})')
    axes[1].grid(alpha=0.3)

    # Rotation against eccentricity. Radial orbits do not rotate, by construction, so
    # the two should trade off against each other.
    axes[2].scatter(sample['ecc'], sample['V_phi'], s=25, alpha=0.7)
    axes[2].axhline(0, color='red', linestyle='--')
    axes[2].set_xlabel('eccentricity')
    axes[2].set_ylabel(r'$V_\phi$ (km/s)')
    axes[2].set_title('Rotation vs orbit shape')
    axes[2].grid(alpha=0.3)

    save_figure(figure, 'rotation.png')

    counter = (sample['V_phi'] < 0).sum()
    print(f'    median V_phi = {sample["V_phi"].median():+.1f} km/s,'
          f' {counter} clusters counter-rotating')
    print('    least rotating (closest to zero, i.e. not joining in):')
    for _, row in sample.assign(a=sample['V_phi'].abs()).nsmallest(5, 'a').iterrows():
        print(f'      {row["ID"]:<12s} V_phi = {row["V_phi"]:+7.1f} km/s,'
              f' ecc = {row["ecc"]:.2f}, R_gc = {row["R_gc"]:.1f} kpc')


def exploratory_analysis(merged):
    """Produce the three exploratory figures."""
    print('\n' + '=' * 76)
    print('2. EXPLORATORY ANALYSIS')
    print('=' * 76)

    plot_age_metallicity(merged)
    plot_spatial_kinematic(merged)
    plot_structural_properties(merged)
    plot_rotation(merged)
