"""
Shared paths and constants.

`save_figure` lives here rather than in either analysis module because both of them
write figures, and this is where FIGURE_DIR is defined.
"""

from pathlib import Path

import matplotlib.pyplot as plt

# Paths resolve relative to this file so the code runs from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
FIGURE_DIR = PROJECT_ROOT / 'figures'

DATA_FILEPATHS = {
    'HarrisPartI': DATA_DIR / 'HarrisPartI.csv',
    'HarrisPartIII': DATA_DIR / 'HarrisPartIII.csv',
    'Krause21': DATA_DIR / 'Krause21.csv',
    'vandenBerg': DATA_DIR / 'vandenBerg_table2.csv',
    'labels': DATA_DIR / 'BelokurovKravtsov24_tableD1.csv',
    'orbits': DATA_DIR / 'Baumgardt_orbits.txt',
}

# Cross-validation settings, shared so every reported score uses the same protocol.
RANDOM_SEED = 0
N_FOLDS = 5


def save_figure(figure, filename):
    """Write a figure into figures/."""
    FIGURE_DIR.mkdir(exist_ok=True)
    path = FIGURE_DIR / filename
    figure.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(figure)
    print(f'    saved {path.relative_to(PROJECT_ROOT)}')
