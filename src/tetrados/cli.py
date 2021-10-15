"""
This module contains a script for using tetrados from the command line.
"""
import warnings

import click
from ruamel.yaml.error import MantissaNoDotYAML1_1Warning

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="tetrados")
warnings.filterwarnings("ignore", category=FutureWarning, module="scipy")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", MantissaNoDotYAML1_1Warning)

zero_weighted_type = click.Choice(["keep", "drop", "prefer"], case_sensitive=False)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("vasprun")
@click.option(
    "-z",
    "--zero-weighted-kpoints",
    help="how to process zero-weighted k-points",
    type=zero_weighted_type,
    default="prefer",
    show_default=True,
)
@click.option("--symprec", help="symprec override (default is SYMPREC from vasprun.xml")
@click.option(
    "--time-reversal/--no-time-reversal",
    default=True,
    help="include time reversal symmetry",
)
@click.option(
    "--estep", default=0.005, type=float, show_default=True, help="energy step in eV"
)
@click.option("-p", "--plot", default=False, is_flag=True, help="plot new vs old dos")
def tetrados(vasprun, zero_weighted_kpoints, symprec, time_reversal, estep, plot):
    """Generate a tetrahedron DOS from a normal DOS."""
    import sys

    import numpy as np
    from pymatgen.io.vasp import Vasprun

    from tetrados.bandstructure import get_band_structure
    from tetrados.kpoints import (
        get_kpoint_mapping,
        get_kpoints_from_bandstructure,
        get_kpoints_tetrahedral,
        get_mesh_from_kpoint_diff,
    )
    from tetrados.symmetry import expand_bandstructure
    from tetrados.tetrahedron import TetrahedralBandStructure

    vasprun = Vasprun(vasprun)

    # smart symprec; use VASP if not set on command line
    if symprec is None:
        symprec = vasprun.parameters["SYMPREC"]
    else:
        symprec = float(symprec)

    band_structure = get_band_structure(vasprun, zero_weighted=zero_weighted_kpoints)
    orig_kpoints = get_kpoints_from_bandstructure(band_structure)
    mesh_dim, is_shifted = get_mesh_from_kpoint_diff(orig_kpoints)

    if is_shifted:
        click.echo("tetrados does not support non Gamma centred meshes")
        sys.exit()

    mesh_dim = mesh_dim.round(0).astype(int)

    # get tetrahedron k-points
    (
        _,
        _,
        tet_kpts,
        ir_kpts_idx,
        ir_to_full_idx,
        tetrahedra,
        *ir_tetrahedra_info,
    ) = get_kpoints_tetrahedral(
        mesh_dim,
        vasprun.final_structure,
        symprec=symprec,
        time_reversal_symmetry=time_reversal,
    )

    # desymmetrize original band structure onto mesh covering full BZ
    band_structure = expand_bandstructure(
        band_structure, symprec=symprec, time_reversal=time_reversal
    )

    # order expanded k-points in the same order as tetrahedron k-points
    expanded_kpoints = get_kpoints_from_bandstructure(band_structure)
    expanded_energies = band_structure.bands

    sort_idx = get_kpoint_mapping(tet_kpts, expanded_kpoints)
    tet_energies = {s: e[:, sort_idx] for s, e in expanded_energies.items()}

    tetrahedral_band_structure = TetrahedralBandStructure.from_data(
        tet_energies,
        tet_kpts,
        tetrahedra,
        vasprun.final_structure,
        ir_kpts_idx,
        ir_to_full_idx,
        *ir_tetrahedra_info,
    )

    emin = np.min([np.min(spin_eners) for spin_eners in expanded_energies.values()])
    emax = np.max([np.max(spin_eners) for spin_eners in expanded_energies.values()])
    epoints = int(round((emax - emin) / estep))
    energies = np.linspace(emin, emax, epoints)

    _, dos = tetrahedral_band_structure.get_density_of_states(energies, sum_spins=True)

    data = np.vstack([energies, dos])
    np.savetxt("tetdos.dat", data.T, header="energy[eV] dos[states/eV]")
    click.echo("saved dos to tetdos.dat")

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        new_energies = energies - vasprun.efermi
        new_mask = (new_energies > -6.1) & (new_energies < 6.1)
        ax.plot(new_energies[new_mask], dos[new_mask], label="tetrados")

        cdos = vasprun.complete_dos
        old_energies = cdos.energies - vasprun.efermi
        old_mask = (old_energies > -6.1) & (old_energies < 6.1)
        old_dos = cdos.get_densities()

        ax.plot(old_energies[old_mask], old_dos[old_mask], label="VASP")
        ax.set(xlim=(-6, 6), xlabel="$E$ - $E_{F}$ (eV)", ylabel="DOS")
        ax.margins(y=0.1)
        ax.legend()
        plt.show()


if __name__ == "__main__":
    tetrados()
