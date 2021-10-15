import numpy as np
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.io.vasp import Vasprun

from tetrados.settings import zero_weighted_kpoints


def get_band_structure(
    vasprun: Vasprun, zero_weighted: str = zero_weighted_kpoints
) -> BandStructure:
    """
    Get a band structure from a Vasprun object.

    This can ensure that if the calculation contains zero-weighted k-points then the
    weighted k-points will be discarded (helps with hybrid calculations).

    Also ensures that the Fermi level is set correctly.

    Args:
        vasprun: A vasprun object.
        zero_weighted: How to handle zero-weighted k-points if they are present in the
            calculation. Options are:
            - "keep": Keep zero-weighted k-points in the band structure.
            - "drop": Drop zero-weighted k-points, keeping only the weighted k-points.
            - "prefer": Drop weighted-kpoints if zero-weighted k-points are present
              in the calculation (useful for cheap hybrid calculations).

    Returns:
        A band structure.
    """
    # first check if Fermi level crosses a band
    k_idx = get_zero_weighted_kpoint_indices(vasprun, mode=zero_weighted)
    kpoints = np.array(vasprun.actual_kpoints)[k_idx]

    projections = {}
    eigenvalues = {}
    for spin, spin_eigenvalues in vasprun.eigenvalues.items():
        # discard weight and set shape nbands, nkpoints
        eigenvalues[spin] = spin_eigenvalues[k_idx, :, 0].transpose(1, 0)

        if vasprun.projected_eigenvalues:
            # is nkpoints, nbands, nion, norb; we need nbands, nkpoints, norb, nion
            spin_projections = vasprun.projected_eigenvalues[spin]
            projections[spin] = spin_projections[k_idx].transpose(1, 0, 3, 2)

    return BandStructure(
        kpoints,
        eigenvalues,
        vasprun.final_structure.lattice.reciprocal_lattice,
        efermi=vasprun.efermi,
        structure=vasprun.final_structure,
        projections=projections,
    )


def get_zero_weighted_kpoint_indices(vasprun: Vasprun, mode: str) -> np.ndarray:
    """
    Get zero weighted k-point k-point indices from a vasprun.

    If the calculation doesn't contain zero-weighted k-points, then the indices of
    all the k-points will be returned. Alternatively, if the calculation contains
    a mix of weighted and zero-weighted k-points, then only the indices of the
    zero-weighted k-points will be returned.

    Args:
        vasprun:  A vasprun object.
        mode: How to handle zero-weighted k-points if they are present in the
            calculation. Options are:
            - "keep": Keep zero-weighted k-points in the band structure.
            - "drop": Drop zero-weighted k-points, keeping only the weighted k-points.
            - "prefer": Drop weighted-kpoints if zero-weighted k-points are present
              in the calculation (useful for cheap hybrid calculations).

    Returns:
        The indices of the valid k-points.
    """
    weights = np.array(vasprun.actual_kpoints_weights)
    is_zero_weight = weights == 0

    if mode not in ("prefer", "drop", "keep"):
        raise ValueError(f"Unrecognised zero-weighted k-point mode: {mode}")

    if mode == "prefer" and np.any(is_zero_weight):
        return np.where(is_zero_weight)[0]
    elif mode == "drop" and np.any(~is_zero_weight):
        return np.where(~is_zero_weight)[0]
    else:
        return np.arange(len(weights))
