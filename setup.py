from pathlib import Path

from setuptools import find_packages, setup

module_dir = Path(__file__).resolve().parent

with open(module_dir / "README.md") as f:
    long_desc = f.read()

setup(
    name="tetrados",
    description="generate tetrahedron density of states",
    long_description=long_desc,
    use_scm_version={"version_scheme": "python-simplified-semver"},
    setup_requires=["setuptools_scm"],
    long_description_content_type="text/markdown",
    url="https://github.com/utf/tetrados",
    author="Alex Ganose",
    author_email="alexganose@gmail.com",
    license="MIT",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"tetrados": ["py.typed"]},
    zip_safe=False,
    include_package_data=True,
    install_requires=[
        "setuptools",
        "click," "pymatgen",
        "numpy",
        "spglib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Operating System :: OS Independent",
        "Topic :: Other/Nonlisted Topic",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "tetrados = tetrados.cli:tetrados",
        ]
    },
)
