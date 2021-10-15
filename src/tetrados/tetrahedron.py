import logging
import time
from typing import Dict, Optional

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Spin
from pymatgen.util.coord import pbc_diff

"""
     6-------7
    /|      /|
   / |     / |
  4-------5  |
  |  2----|--3
  | /     | /
  |/      |/
  0-------1

 i: vec        neighbours
 0: O          1, 2, 4
 1: a          0, 3, 5
 2: b          0, 3, 6
 3: a + b      1, 2, 7
 4: c          0, 5, 6
 5: c + a      1, 4, 7
 6: c + b      2, 4, 7
 7: c + a + b  3, 5, 6
"""

_main_diagonals = (
    (1, 1, 1),  # 0-7
    (-1, 1, 1),  # 1-6
    (1, -1, 1),  # 2-5
    (1, 1, -1),  # 3-4
)

_tetrahedron_vectors = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
)

logger = logging.getLogger(__name__)


def get_main_diagonal(reciprocal_lattice: np.ndarray) -> int:
    # want a list of tetrahedra as (k1, k2, k3, k4); as per the Bloechl paper,
    # the order of the k-points is irrelevant and therefore they should be ordered
    # in increasing number
    # All tetrahedrons for a specific submesh will share one common diagonal.
    # To minimise interpolaiton distances, we should choose the shortest
    # diagonal
    diagonal_lengths = []
    for diagonal in _main_diagonals:
        length = np.linalg.norm(reciprocal_lattice @ diagonal)
        diagonal_lengths.append(length)

    return int(np.argmin(diagonal_lengths))


def get_relative_tetrahedron_vertices(reciprocal_lattice):
    shortest_index = get_main_diagonal(reciprocal_lattice)

    if shortest_index == 0:
        pairs = ((1, 3), (1, 5), (2, 3), (2, 6), (4, 5), (4, 6))
        main = (0, 7)

    elif shortest_index == 1:
        pairs = ((0, 2), (0, 4), (2, 3), (3, 7), (4, 5), (5, 7))
        main = (1, 6)

    elif shortest_index == 2:
        pairs = ((0, 1), (0, 4), (1, 3), (3, 7), (4, 6), (6, 7))
        main = (2, 5)

    elif shortest_index == 3:
        pairs = ((0, 1), (0, 2), (1, 5), (2, 6), (5, 7), (6, 7))
        main = (3, 4)

    else:
        assert False

    tetras = np.sort([main + x for x in pairs])
    return _tetrahedron_vectors[tetras]


def get_tetrahedra(
    reciprocal_lattice: np.ndarray,
    grid_address: np.ndarray,
    mesh: np.ndarray,
    grid_address_mapping,
):
    tetrahedron_vertices = get_relative_tetrahedron_vertices(reciprocal_lattice)

    grid_order = [1, mesh[0], mesh[0] * mesh[1]]

    all_grid_points = np.repeat(grid_address, [24] * len(grid_address), axis=0)
    all_vertices = np.tile(tetrahedron_vertices, (len(grid_address), 1, 1))
    points = all_grid_points.reshape(all_vertices.shape) + all_vertices

    # fancy magic from phonopy to get neighboring indices given relative coordinates
    tetrahedra = np.dot(points % mesh, grid_order)

    ir_tetrahedra_vertices = grid_address_mapping[tetrahedra]
    _, ir_tetrahedra_idx, ir_tetrahedra_to_full_idx, ir_weights = np.unique(
        np.sort(ir_tetrahedra_vertices),
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )

    return tetrahedra, ir_tetrahedra_idx, ir_tetrahedra_to_full_idx, ir_weights


class TetrahedralBandStructure:
    def __init__(
        self,
        energies: Dict[Spin, np.ndarray],
        kpoints: np.ndarray,
        ir_kpoints_idx: np.ndarray,
        ir_kpoint_mapping: np.ndarray,
        ir_kpoint_weights: np.ndarray,
        tetrahedra: Dict[Spin, np.ndarray],
        ir_tetrahedra: Dict[Spin, np.ndarray],
        ir_tetrahedra_energies: Dict[Spin, np.ndarray],
        ir_tetrahedra_idx: np.ndarray,
        ir_tetrahedra_to_full_idx: np.ndarray,
        ir_tetrahedra_weights: np.ndarray,
        e21: Dict[Spin, np.ndarray],
        e31: Dict[Spin, np.ndarray],
        e41: Dict[Spin, np.ndarray],
        e32: Dict[Spin, np.ndarray],
        e42: Dict[Spin, np.ndarray],
        e43: Dict[Spin, np.ndarray],
        max_tetrahedra_energies: Dict[Spin, np.ndarray],
        min_tetrahedra_energies: Dict[Spin, np.ndarray],
        cross_section_weights: Dict[Spin, np.ndarray],
        tetrahedron_volume: float,
        weights_cache: Optional[Dict[Spin, np.ndarray]] = None,
        weights_mask_cache: Optional[Dict[Spin, np.ndarray]] = None,
        energies_cache: Optional[Dict[Spin, np.ndarray]] = None,
    ):
        self.energies = energies
        self.kpoints = kpoints
        self.ir_kpoints_idx = ir_kpoints_idx
        self.ir_kpoint_mapping = ir_kpoint_mapping
        self.ir_kpoint_weights = ir_kpoint_weights
        self.tetrahedra = tetrahedra
        self.ir_tetrahedra = ir_tetrahedra
        self.ir_tetrahedra_energies = ir_tetrahedra_energies
        self.ir_tetrahedra_idx = ir_tetrahedra_idx
        self.ir_tetrahedra_to_full_idx = ir_tetrahedra_to_full_idx
        self.ir_tetrahedra_weights = ir_tetrahedra_weights
        self.e21 = e21
        self.e31 = e31
        self.e41 = e41
        self.e32 = e32
        self.e42 = e42
        self.e43 = e43
        self.max_tetrahedra_energies = max_tetrahedra_energies
        self.min_tetrahedra_energies = min_tetrahedra_energies
        self.cross_section_weights = cross_section_weights
        self._tetrahedron_volume = tetrahedron_volume
        self._weights_cache = {} if weights_cache is None else weights_cache
        self._weights_mask_cache = (
            {} if weights_mask_cache is None else weights_mask_cache
        )
        self._energies_cache = {} if energies_cache is None else energies_cache

        self._ir_weights_shape = {
            s: (len(energies[s]), len(ir_kpoints_idx)) for s in energies
        }

    @classmethod
    def from_data(
        cls,
        energies: Dict[Spin, np.ndarray],
        kpoints: np.ndarray,
        tetrahedra: np.ndarray,
        structure: Structure,
        ir_kpoints_idx: np.ndarray,
        ir_kpoint_mapping: np.ndarray,
        ir_tetrahedra_idx: Optional[np.ndarray] = None,
        ir_tetrahedra_to_full_idx: Optional[np.ndarray] = None,
        ir_tetrahedra_weights: Optional[np.ndarray] = None,
    ):
        logger.info("Initializing tetrahedron band structure")
        t0 = time.perf_counter()

        tparams = (ir_tetrahedra_idx, ir_tetrahedra_to_full_idx, ir_tetrahedra_weights)
        if len({x is None for x in tparams}) != 1:
            raise ValueError(
                "Either all or none of ir_tetrahedra_idx, ir_tetrahedra_to_full_idx and"
                " ir_tetrahedra_weights should be set."
            )

        if ir_tetrahedra_idx is None:
            ir_tetrahedra_idx = np.arange(len(kpoints))
            ir_tetrahedra_to_full_idx = np.ones_like(ir_tetrahedra_idx)
            ir_tetrahedra_weights = np.ones_like(ir_tetrahedra_idx)

        ir_tetrahedra_to_full_idx = ir_tetrahedra_to_full_idx
        ir_kpoints_idx = ir_kpoints_idx
        ir_kpoint_mapping = ir_kpoint_mapping

        _, ir_kpoint_weights = np.unique(ir_kpoint_mapping, return_counts=True)

        # need to keep track of full tetrahedra to recover full k-point indices
        # when calculating scattering rates (i.e., k-k' is symmetry inequivalent).
        full_tetrahedra, _ = process_tetrahedra(tetrahedra, energies)

        # store irreducible tetrahedra and use energies to calculate diffs and min/maxes
        ir_tetrahedra, ir_tetrahedra_energies = process_tetrahedra(
            tetrahedra[ir_tetrahedra_idx], energies
        )

        # the remaining properties are given for each irreducible tetrahedra
        (e21, e31, e41, e32, e42, e43) = get_tetrahedra_energy_diffs(
            ir_tetrahedra_energies
        )

        (
            max_tetrahedra_energies,
            min_tetrahedra_energies,
        ) = get_max_min_tetrahedra_energies(ir_tetrahedra_energies)

        cross_section_weights = get_tetrahedra_cross_section_weights(
            structure.lattice.reciprocal_lattice.matrix,
            kpoints,
            ir_tetrahedra,
            e21,
            e31,
            e41,
        )

        tetrahedron_volume = 1 / len(tetrahedra)

        logger.info(f"  time: {time.perf_counter() - t0:.4f} s")
        return cls(
            energies,
            kpoints,
            ir_kpoints_idx,
            ir_kpoint_mapping,
            ir_kpoint_weights,
            full_tetrahedra,
            ir_tetrahedra,
            ir_tetrahedra_energies,
            ir_tetrahedra_idx,
            ir_tetrahedra_to_full_idx,
            ir_tetrahedra_weights,
            e21,
            e31,
            e41,
            e32,
            e42,
            e43,
            max_tetrahedra_energies,
            min_tetrahedra_energies,
            cross_section_weights,
            tetrahedron_volume,
        )

    def get_intersecting_tetrahedra(self, spin, energy, band_idx=None):

        max_energies = self.max_tetrahedra_energies[spin]
        min_energies = self.min_tetrahedra_energies[spin]

        if band_idx is not None:
            mask = np.full_like(max_energies, False, dtype=bool)
            mask[band_idx] = True

            return (min_energies < energy) & (max_energies > energy) & mask

        else:
            return (min_energies < energy) & (max_energies > energy)

    def get_density_of_states(
        self,
        energies=None,
        integrand=None,
        sum_spins=False,
        band_idx=None,
        use_cached_weights=False,
    ):
        if energies is None:
            from tetrados.settings import dos_estep

            min_e = np.min([np.min(e) for e in self.energies.values()])
            max_e = np.max([np.max(e) for e in self.energies.values()])
            energies = np.arange(min_e, max_e, dos_estep)

        dos = {}
        for spin in self.energies.keys():
            if isinstance(integrand, dict):
                # integrand given for each spin channel
                spin_integrand = integrand[spin]
            else:
                spin_integrand = integrand

            if isinstance(band_idx, dict):
                # band indices given for each spin channel
                spin_band_idx = band_idx[spin]
            else:
                spin_band_idx = band_idx

            if spin_integrand is not None:
                if spin_integrand.shape[:2] != self.energies[spin].shape:
                    raise ValueError(
                        "Unexpected integrand shape, should be (nbands, nkpoints, ...)"
                    )

                nbands = len(spin_integrand)
                integrand_shape = spin_integrand.shape[2:]
                n_ir_kpoints = len(self.ir_kpoints_idx)
                new_integrand = np.zeros((nbands, n_ir_kpoints) + integrand_shape)

                flat_k = np.tile(self.ir_kpoint_mapping, nbands)
                flat_b = np.repeat(np.arange(nbands), len(self.ir_kpoint_mapping))
                flat_integrand = spin_integrand.reshape((-1,) + integrand_shape)
                # flat_integrand = spin_integrand.reshape(-1, 3, 3)

                # sum integrand at all symmetry equivalent points, new_integrand
                # has shape (nbands, n_ir_kpoints)
                np.add.at(new_integrand, (flat_b, flat_k), flat_integrand)
                spin_integrand = new_integrand

            _, dos[spin] = self.get_spin_density_of_states(
                spin,
                energies,
                integrand=spin_integrand,
                band_idx=spin_band_idx,
                use_cached_weights=use_cached_weights,
            )

        if sum_spins:
            if Spin.down in dos:
                dos = dos[Spin.up] + dos[Spin.down]
            else:
                dos = dos[Spin.up]

        return energies, dos

    def get_spin_density_of_states(
        self,
        spin,
        energies,
        integrand=None,
        band_idx=None,
        use_cached_weights=False,
    ):
        # integrand should have the shape (nbands, n_ir_kpts, ...)
        # the integrand should have been summed at all equivalent k-points
        # TODO: add support for variable shaped integrands
        if integrand is None:
            dos = np.zeros_like(energies)
        else:
            integrand_shape = integrand.shape[2:]
            dos = np.zeros((len(energies),) + integrand_shape)

        if use_cached_weights:
            if self._weights_cache is None:
                raise ValueError("No integrand have been cached")

            all_weights = self._weights_cache[spin]
            all_weights_mask = self._weights_mask_cache[spin]
            energies = self._energies_cache[spin]
        else:
            all_weights = []
            all_weights_mask = []

        nbands = len(self.energies[spin])
        kpoint_multiplicity = np.tile(self.ir_kpoint_weights, (nbands, 1))

        if band_idx is not None and integrand is not None:
            integrand = integrand[band_idx]

        if band_idx is not None and integrand is None:
            kpoint_multiplicity = kpoint_multiplicity[band_idx]

        energies_iter = list(enumerate(energies))

        for i, energy in energies_iter:
            if use_cached_weights:
                weights = all_weights[i]
                weights_mask = all_weights_mask[i]
            else:
                weights = self.get_energy_dependent_integration_weights(spin, energy)
                weights_mask = weights != 0
                all_weights.append(weights)
                all_weights_mask.append(weights_mask)

            if band_idx is not None:
                weights = weights[band_idx]
                weights_mask = weights_mask[band_idx]

            if integrand is None:
                dos[i] = np.sum(
                    weights[weights_mask] * kpoint_multiplicity[weights_mask]
                )

            else:
                # expand weights to match the dimensions of the integrand
                expand_axis = [1 + i for i in range(len(integrand.shape[2:]))]
                expand_weights = np.expand_dims(weights[weights_mask], axis=expand_axis)

                # don't need to include the k-point multiplicity as this is included by
                # pre-summing the integrand at symmetry equivalent points
                dos[i] = np.sum(expand_weights * integrand[weights_mask], axis=0)

        if not use_cached_weights:
            self._weights_cache[spin] = np.array(all_weights)
            self._weights_mask_cache[spin] = np.array(all_weights_mask)
            self._energies_cache[spin] = energies

        return energies, np.asarray(dos)

    def get_energy_dependent_integration_weights(self, spin, energy):
        integration_weights = np.zeros(self._ir_weights_shape[spin])
        tetrahedra_mask = self.get_intersecting_tetrahedra(spin, energy)

        if not np.any(tetrahedra_mask):
            return integration_weights

        energies = self.ir_tetrahedra_energies[spin][tetrahedra_mask]
        e21 = self.e21[spin][tetrahedra_mask]
        e31 = self.e31[spin][tetrahedra_mask]
        e41 = self.e41[spin][tetrahedra_mask]
        e32 = self.e32[spin][tetrahedra_mask]
        e42 = self.e42[spin][tetrahedra_mask]
        e43 = self.e43[spin][tetrahedra_mask]

        cond_a_mask = (energies[:, 0] < energy) & (energy < energies[:, 1])
        cond_b_mask = (energies[:, 1] <= energy) & (energy < energies[:, 2])
        cond_c_mask = (energies[:, 2] <= energy) & (energy < energies[:, 3])

        ee1 = energy - energies[:, 0]
        ee2 = energy - energies[:, 1]
        ee3 = energy - energies[:, 2]
        e2e = energies[:, 1] - energy
        e3e = energies[:, 2] - energy
        e4e = energies[:, 3] - energy

        kpoints_idx = self.ir_tetrahedra[spin][tetrahedra_mask]
        ir_kpoints_idx = self.ir_kpoint_mapping[kpoints_idx]

        # calculate the integrand for each vertices
        vert_weights = np.zeros_like(energies)
        vert_weights[cond_a_mask] = _get_energy_dependent_weight_a(
            ee1[cond_a_mask],
            e2e[cond_a_mask],
            e3e[cond_a_mask],
            e4e[cond_a_mask],
            e21[cond_a_mask],
            e31[cond_a_mask],
            e41[cond_a_mask],
        )

        vert_weights[cond_b_mask] = _get_energy_dependent_weight_b(
            ee1[cond_b_mask],
            ee2[cond_b_mask],
            e3e[cond_b_mask],
            e4e[cond_b_mask],
            e31[cond_b_mask],
            e41[cond_b_mask],
            e32[cond_b_mask],
            e42[cond_b_mask],
        )

        vert_weights[cond_c_mask] = _get_energy_dependent_weight_c(
            ee1[cond_c_mask],
            ee2[cond_c_mask],
            ee3[cond_c_mask],
            e4e[cond_c_mask],
            e41[cond_c_mask],
            e42[cond_c_mask],
            e43[cond_c_mask],
        )

        # finally, get the integrand for each ir_kpoint by summing over all
        # tetrahedra and multiplying by the tetrahedra multiplicity and
        # tetrahedra weight; Finally, divide by the k-point multiplicity
        # to get the final weight
        band_idx, tetrahedra_idx = np.where(tetrahedra_mask)

        # include tetrahedra multiplicity
        vert_weights *= self.ir_tetrahedra_weights[tetrahedra_idx][:, None]

        flat_ir_kpoints = np.ravel(ir_kpoints_idx)
        flat_ir_weights = np.ravel(vert_weights)
        flat_bands = np.repeat(band_idx, 4)

        # sum integrand, note this sums in place and is insanely fast
        np.add.at(integration_weights, (flat_bands, flat_ir_kpoints), flat_ir_weights)
        integration_weights *= (
            self._tetrahedron_volume / self.ir_kpoint_weights[None, :]
        )

        return integration_weights


def _get_energy_dependent_weight_a(ee1, e2e, e3e, e4e, e21, e31, e41):
    c = ee1 ** 2 / (e21 * e31 * e41)
    i1 = c * (e2e / e21 + e3e / e31 + e4e / e41)
    i2 = c * (ee1 / e21)
    i3 = c * (ee1 / e31)
    i4 = c * (ee1 / e41)
    return np.stack([i1, i2, i3, i4], axis=1)


def _get_energy_dependent_weight_b(ee1, ee2, e3e, e4e, e31, e41, e32, e42):
    c = (ee1 * e4e) / (e31 * e41 * e42)
    x = e3e / e31
    y = e4e / e42
    z = ee2 / (e32 * e42)
    zx = z * x
    k = ee1 / e31
    n = ee2 / e42

    i1 = c * (x + e4e / e41) + z * x ** 2
    i2 = c * y + zx * (e3e / e32 + y)
    i3 = c * k + zx * (k + ee2 / e32)
    i4 = c * (ee1 / e41 + n) + zx * n
    return np.stack([i1, i2, i3, i4], axis=1)


def _get_energy_dependent_weight_c(ee1, ee2, ee3, e4e, e41, e42, e43):
    c = e4e ** 2 / (e41 * e42 * e43)
    i1 = c * e4e / e41
    i2 = c * e4e / e42
    i3 = c * e4e / e43
    i4 = c * (ee1 / e41 + ee2 / e42 + ee3 / e43)
    return np.stack([i1, i2, i3, i4], axis=1)


def process_tetrahedra(tetrahedra, energies):
    all_tetrahedra = {}
    all_tetrahedra_energies = {}

    for spin, spin_energies in energies.items():
        data_shape = (len(spin_energies),) + tetrahedra.shape
        spin_tetrahedra = np.zeros(data_shape, dtype=int)
        spin_tetrahedra_energies = np.zeros(data_shape)

        for band_idx, band_energies in enumerate(spin_energies):
            band_tetrahedra_energies = band_energies[tetrahedra]

            sort_idx = np.argsort(band_tetrahedra_energies, axis=1)
            spin_tetrahedra_energies[band_idx, ...] = np.take_along_axis(
                band_tetrahedra_energies, sort_idx, axis=1
            )
            spin_tetrahedra[band_idx, ...] = np.take_along_axis(
                tetrahedra, sort_idx, axis=1
            )

        all_tetrahedra[spin] = spin_tetrahedra
        all_tetrahedra_energies[spin] = spin_tetrahedra_energies

    return all_tetrahedra, all_tetrahedra_energies


def get_tetrahedra_energy_diffs(tetrahedra_energies):
    e21 = {}
    e31 = {}
    e41 = {}
    e32 = {}
    e42 = {}
    e43 = {}

    for spin, s_tetrahedra_energies in tetrahedra_energies.items():
        # each energy difference has the shape nbands, ntetrahedra
        e21[spin] = s_tetrahedra_energies[:, :, 1] - s_tetrahedra_energies[:, :, 0]
        e31[spin] = s_tetrahedra_energies[:, :, 2] - s_tetrahedra_energies[:, :, 0]
        e41[spin] = s_tetrahedra_energies[:, :, 3] - s_tetrahedra_energies[:, :, 0]
        e32[spin] = s_tetrahedra_energies[:, :, 2] - s_tetrahedra_energies[:, :, 1]
        e42[spin] = s_tetrahedra_energies[:, :, 3] - s_tetrahedra_energies[:, :, 1]
        e43[spin] = s_tetrahedra_energies[:, :, 3] - s_tetrahedra_energies[:, :, 2]

    return e21, e31, e41, e32, e42, e43


def get_max_min_tetrahedra_energies(tetrahedra_energies):
    max_tetrahedra_energies = {}
    min_tetrahedra_energies = {}

    for spin, s_tetrahedra_energies in tetrahedra_energies.items():
        max_tetrahedra_energies[spin] = np.max(s_tetrahedra_energies, axis=2)
        min_tetrahedra_energies[spin] = np.min(s_tetrahedra_energies, axis=2)

    return max_tetrahedra_energies, min_tetrahedra_energies


def get_tetrahedra_cross_section_weights(
    reciprocal_lattice, kpoints, tetrahedra, e21, e31, e41
):
    # weight (b) defined by equation 3.4 in https://doi.org/10.1002/pssb.2220540211
    # this weight is not the Bloechl integrand but a scaling needed to obtain the
    # DOS directly from the tetrahedron cross section
    cross_section_weights = {}

    # volume is 6 * the volume of one tetrahedron
    volume = np.linalg.det(reciprocal_lattice) / len(kpoints)
    for spin, s_tetrahedra in tetrahedra.items():
        tetrahedra_kpoints = kpoints[s_tetrahedra]

        k1 = pbc_diff(tetrahedra_kpoints[:, :, 1], tetrahedra_kpoints[:, :, 0])
        k2 = pbc_diff(tetrahedra_kpoints[:, :, 2], tetrahedra_kpoints[:, :, 0])
        k3 = pbc_diff(tetrahedra_kpoints[:, :, 3], tetrahedra_kpoints[:, :, 0])

        k1_cart = np.dot(k1, reciprocal_lattice)
        k2_cart = np.dot(k2, reciprocal_lattice)
        k3_cart = np.dot(k3, reciprocal_lattice)

        contragradient = np.stack(
            [
                np.cross(k2_cart, k3_cart) / volume,
                np.cross(k3_cart, k1_cart) / volume,
                np.cross(k1_cart, k2_cart) / volume,
            ],
            axis=2,
        )

        energies = np.stack([e21[spin], e31[spin], e41[spin]], axis=2)
        b_vector = np.sum(contragradient * energies[..., None], axis=2)

        cross_section_weights[spin] = 1 / np.linalg.norm(b_vector, axis=2)

    return cross_section_weights
