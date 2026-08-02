"""Descriptor-space uncertainty decomposition for unpaired virtual staining.

Implements the φ_struct pipeline of `kidney_ood_data_plan.md` §5 and the error
decomposition of `uncertainty_strategy.md` §2.1:

    E_m‖φ(G_m(x)) − φ(y)‖²  =  Var(x)  +  bias²(x)  +  floor

Everything here averages in **descriptor space**, never pixel space — averaging
GAN outputs pixel-wise blurs into a non-image, which is why this package exists
alongside (not instead of) the per-pixel `uncertainty.py`.

Modules
-------
descriptors  φ_struct: the 6-component structural vector
regions      1–2 mm region grid derived from tiles_metadata.csv
ensemble     per-member φ, then μ and Var across members
decompose    law of total variance over the (fold × seed) grid
floor        biological-floor covariance, bracketed from both sides
whiten       Ledoit–Wolf shrinkage, Mahalanobis norm, bias² = observed² − d
"""

from uncertainty_phi.descriptors import (  # noqa: F401
    PHI_NAMES,
    PHI_REFERENCE,
    phi_struct,
    betti,
    clean_mask,
    collagen_fraction,
    regional_dispersion,
    lumen_tissue_fraction,
    he_bright,
    he_tissue_footprint,
)

__all__ = [
    "PHI_NAMES",
    "PHI_REFERENCE",
    "phi_struct",
    "betti",
    "clean_mask",
    "collagen_fraction",
    "regional_dispersion",
    "lumen_tissue_fraction",
    "he_bright",
    "he_tissue_footprint",
]
