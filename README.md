# tetrados

tetrados is a tool to generate a density of states using the linear tetrahedron method
from a band structure. Currently, only VASP calculations are supported. Calculations
must be performed on a uniform Gamma-centred k-point mesh.

## Installation

tetrados is written in python. Clone the repository and install tetrados from source.
We recommend installing inside a conda environment.

```bash
git clone https://github.com/utf/tetrados.git
cd tetrados
pip install .
```

tetrados depends on:

- pymatgen
- numpy
- spglib

## Usage

The only required input is a vasprun.xml file. The following will output a file named
`tetdos.dat` in the current directory with the tetrahedron density of states.

```bash
tetrados vasprun.xml
```

tetrados can handle zero weighted k-points using the ` --zero-weighted-kpoints` option.
There are three modes:

- `prefer` (default): Drop weighted-kpoints if zero-weighted k-points are present in the
  calculation (useful for cheap hybrid calculations).
- `drop`: Drop zero-weighted k-points, keeping only the weighted k-points.
- `keep`: Keep both zero-weighted k-points and weighted k-points in the band
  structure (note: this likely won't work as the mesh will no longer be uniform).

The full set of options can be printed using:

```bash
tetrados --help
```

## License

tetrados is released under the MIT license; the full text can be found [here][license].

## Acknowledgements

tetrados was designed and developed by Alex Ganose.

[license]: https://raw.githubusercontent.com/utf/tetrados/main/LICENSE
