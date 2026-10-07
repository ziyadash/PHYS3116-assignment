"""
PHYS3116 Assignment - Option 1: Accreted Milky Way Globular Clusters

Entry point. Runs the three stages in order:

    data_prep        load the Harris, Krause and VandenBerg catalogues, reconcile
                     cluster names, merge, attach the reference classifications
    exploratory      descriptive figures, label-free
    method_design    logistic regression on a designed feature vector

Run with:  python src/main.py
Figures are written to figures/. See docs/methodology.md for the method write-up.

Reference for the in-situ/accreted labels:
    Belokurov & Kravtsov (2024), MNRAS 528, 3198 (arXiv:2309.15902), Appendix D.
"""

# Assignment spec
'''
Using data from the Harris Catalogue of Milky Way globular clusters and van den Berg et al. (2013),
use stellar population (metallicities [Fe/H] and ages) and dynamical information to identify potentially
accreted globular clusters. Consider if there are some globular clusters that stand out in the agemetallicity 
relation or that do not rotate with the bulk of other globular clusters.
Consider what the above findings might mean for how the Milky Way formed and how many of its
globular clusters could be accreted and how many may have formed within the Milky Way. Consider
what further tests could be performed to delineate various possible formation scenarios.

'''

from data_prep import prepare_data
from exploratory import exploratory_analysis
from method_design import method_design


def main():
    """Run the full pipeline and return the merged catalogue."""
    merged = prepare_data()
    exploratory_analysis(merged)
    method_design(merged)
    return merged


if __name__ == '__main__':
    main()
