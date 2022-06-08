[![Built Status](https://api.cirrus-ci.com/github/JoseAngelMartinB/robin.svg?branch=main)](https://cirrus-ci.com/github/JoseAngelMartinB/robin)
[![ReadTheDocs](https://readthedocs.org/projects/robin/badge/?version=latest)](https://robin.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/JoseAngelMartinB/robin/main.svg)](https://coveralls.io/r/JoseAngelMartinB/robin)
[![PyPI-Server](https://img.shields.io/pypi/v/robin.svg)](https://pypi.org/project/robin/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/robin.svg)](https://anaconda.org/conda-forge/robin)

<!-- These are examples of badges you might also want to add to your README. Update the URLs accordingly.
[![Built Status](https://api.cirrus-ci.com/github/<USER>/robin.svg?branch=main)](https://cirrus-ci.com/github/<USER>/robin)
[![ReadTheDocs](https://readthedocs.org/projects/robin/badge/?version=latest)](https://robin.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/robin/main.svg)](https://coveralls.io/r/<USER>/robin)
[![PyPI-Server](https://img.shields.io/pypi/v/robin.svg)](https://pypi.org/project/robin/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/robin.svg)](https://anaconda.org/conda-forge/robin)
[![Monthly Downloads](https://pepy.tech/badge/robin/month)](https://pepy.tech/project/robin)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/robin)
[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
-->

![ROBIN (Rail mOBIlity simulatioN) logo](docs/_static/images/logo.png "ROBIN (Rail mOBIlity simulatioN)")

ROBIN (Rail mOBIlity simulatioN)


## Installation

In order to set up the necessary environment:

1. review and uncomment what you need in `environment.yml` and create an environment `robin` with the help of [conda]:
   ```
   conda env create -f environment.yml
   ```
2. activate the new environment with:
   ```
   conda activate robin
   ```

> **_NOTE:_**  The conda environment will have robin installed in editable mode.
> Some changes, e.g. in `setup.cfg`, might require you to run `pip install -e .` again.


Optional and needed only once after `git clone`:

3. install several [pre-commit] git hooks with:
   ```bash
   pre-commit install
   # You might also want to run `pre-commit autoupdate`
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.


Then take a look into the `scripts` and `notebooks` folders.

## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in `environment.yml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yml` for the exact reproduction of your
   environment with:
   ```bash
   conda env export -n robin -f environment.lock.yml --no-builds
   ```
   For multi-OS development, consider using `--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yml` using:
   ```bash
   conda env update -f environment.lock.yml --prune
   ```
## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Guidelines for contributing to this project.
├── Dockerfile              <- Build a docker container with `docker build .`.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data                    <- Directtory for data files
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- [DEPRECATED] Use `python setup.py develop` to install for
│                              development or `python setup.py bdist_wheel` to build.
├── src
│   └── robin               <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `pytest`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

## Known Issues
No known issues at this moment.


## Support
If you have any kind of problem with the program, please feel free to contact José Ángel Martin at JoseAngel.Martin@uclm.es and Ricardo García at Ricardo.Garcia@uclm.es


## Authors

This project is developed by the [MAT](https://blog.uclm.es/grupomat/) and [ORETO](https://www.uclm.es/Home/Misiones/Investigacion/OfertaCientificoTecnica/GruposInvestigacion/DetalleGrupo?idgrupo=75) research groups of the [Escuela Superior de Informática](https://esi.uclm.es) of the [University of Castilla-La Mancha (UCLM)](https://www.uclm.es).
