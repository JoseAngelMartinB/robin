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

<!-- Short description goes here -->

## Installation
Under construction...


## Known Issues
There are no known issues at this moment.


## Support
If you have any kind of problem with the program, please feel free to contact José Ángel Martin at JoseAngel.Martin@uclm.es and Ricardo García at Ricardo.Garcia@uclm.es


## Authors
This project is developed by the [MAT](https://blog.uclm.es/grupomat/) and [ORETO](https://www.uclm.es/Home/Misiones/Investigacion/OfertaCientificoTecnica/GruposInvestigacion/DetalleGrupo?idgrupo=75) research groups of the [Escuela Superior de Informática](https://esi.uclm.es) of the [University of Castilla-La Mancha (UCLM)](https://www.uclm.es).



## Contribution
In order to contribute to this proyect it is necessary to set up the following environment:

1. Create an environment `robin` with the help of [conda](https://anaconda.org):
   ```
   conda env create -f environment.yml
   ```
2. activate the new environment with:
   ```
   conda activate robin
   ```

> **_NOTE:_**  The conda environment will have the package `robin` installed in editable mode.
> Some changes, e.g. in `setup.cfg`, might require you to run `pip install -e .` again.

This project uses pre-commit, please make sure to install it before making any changes:

3. install several [pre-commit] git hooks with:
   ```bash
   pre-commit install
   ```
   It is a good idea to update the hooks to the latest version:
    ```bash
   pre-commit autoupdate
   ```

For more information you can refer to the [contributing guide](CONTRIBUTING.rst).


### Project Organization
```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Guidelines for contributing to this project.
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
