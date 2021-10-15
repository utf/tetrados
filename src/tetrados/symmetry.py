import logging

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import BandStructure

from tetrados.kpoints import get_kpoints_from_bandstructure, get_mesh_from_kpoint_diff
from tetrados.settings import symprec

logger = logging.getLogger(__name__)


def expand_kpoints(
    structure,
    kpoints,
    symprec=symprec,
    return_mapping=False,
    time_reversal=True,
    verbose=True,
):
    if verbose:
        logger.info("Desymmetrizing k-point mesh")

    kpoints = np.array(kpoints).round(8)

    # due to limited input precision of the k-points, the mesh is returned as a float
    mesh, is_shifted = get_mesh_from_kpoint_diff(kpoints)
    status_info = ["Found initial mesh: {:.3f} x {:.3f} x {:.3f}".format(*mesh)]

    if is_shifted:
        shift = np.array([1, 1, 1])
    else:
        shift = np.array([0, 0, 0])

    # to avoid issues to limited input precision, recalculate the input k-points
    # so that the mesh is integer and the k-points are not truncated
    # to a small precision
    addresses = np.rint((kpoints + shift / (mesh * 2)) * mesh)
    mesh = np.rint(mesh)
    kpoints = addresses / mesh - shift / (mesh * 2)

    status_info.append("Integer mesh: {} x {} x {}".format(*map(int, mesh)))

    rotations, translations, is_tr = get_reciprocal_point_group_operations(
        structure, symprec=symprec, time_reversal=time_reversal
    )
    n_ops = len(rotations)
    if verbose:
        status_info.append(f"Using {n_ops} symmetry operations")
        logger.info("\n".join(status_info))

    # rotate all-kpoints
    all_rotated_kpoints = []
    for r in rotations:
        all_rotated_kpoints.append(np.dot(r, kpoints.T).T)
    all_rotated_kpoints = np.concatenate(all_rotated_kpoints)

    # map to first BZ
    all_rotated_kpoints -= np.rint(all_rotated_kpoints)
    all_rotated_kpoints = all_rotated_kpoints.round(8)

    # zone boundary consistent with VASP not with spglib
    all_rotated_kpoints[all_rotated_kpoints == -0.5] = 0.5

    # Find unique points
    unique_rotated_kpoints, unique_idxs = np.unique(
        all_rotated_kpoints, return_index=True, axis=0
    )

    # find integer addresses
    unique_addresses = (unique_rotated_kpoints + shift / (mesh * 2)) * mesh
    unique_addresses -= np.rint(unique_addresses)
    in_uniform_mesh = (np.abs(unique_addresses) < 1e-5).all(axis=1)

    n_mapped = int(np.sum(in_uniform_mesh))
    n_expected = int(np.product(mesh))
    if n_mapped != n_expected:
        raise ValueError(f"Expected {n_expected} points but found {n_mapped}")

    full_kpoints = unique_rotated_kpoints[in_uniform_mesh]
    full_idxs = unique_idxs[in_uniform_mesh]

    if not return_mapping:
        return full_kpoints

    op_mapping = np.floor(full_idxs / len(kpoints)).astype(int)
    kp_mapping = (full_idxs % len(kpoints)).astype(int)

    return full_kpoints, rotations, translations, is_tr, op_mapping, kp_mapping


def get_reciprocal_point_group_operations(
    structure: Structure,
    symprec: float = symprec,
    time_reversal: bool = True,
):
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    if sga.get_symmetry_dataset() is None:
        # sometimes default angle tolerance doesn't work as expected
        sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=-1)

    rotations = sga.get_symmetry_dataset()["rotations"].transpose((0, 2, 1))
    translations = sga.get_symmetry_dataset()["translations"]
    is_tr = np.full(len(rotations), False, dtype=bool)

    if time_reversal:
        rotations = np.concatenate([rotations, -rotations])
        translations = np.concatenate([translations, -translations])
        is_tr = np.concatenate([is_tr, ~is_tr])

        rotations, unique_ops = np.unique(rotations, axis=0, return_index=True)
        translations = translations[unique_ops]
        is_tr = is_tr[unique_ops]

    # put identity first and time-reversal last
    sort_idx = np.argsort(np.abs(rotations - np.eye(3)).sum(axis=(1, 2)) + is_tr * 10)

    return rotations[sort_idx], translations[sort_idx], is_tr[sort_idx]


def expand_bandstructure(bandstructure, symprec=symprec, time_reversal=True):
    kpoints = get_kpoints_from_bandstructure(bandstructure)
    full_kpoints, _, _, _, _, kp_mapping = expand_kpoints(
        bandstructure.structure,
        kpoints,
        symprec=symprec,
        time_reversal=time_reversal,
        return_mapping=True,
    )
    return BandStructure(
        full_kpoints,
        {s: b[:, kp_mapping] for s, b in bandstructure.bands.items()},
        bandstructure.structure.lattice.reciprocal_lattice,
        bandstructure.efermi,
        structure=bandstructure.structure,
    )
