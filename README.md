# SOFA_ZOO
This repository is part of the project "LapGym - An Open Source Framework for Reinforcement Learning in Robot-Assisted Laparoscopic Surgery".
It provides the code for the reinforcement learning experiments as described in the [LapGym paper](https://www.jmlr.org/papers/v24/23-0207.html) for the environments of [sofa_env](https://github.com/ScheiklP/sofa_env).

## Dependencies
* ffmpeg for recording sample videos (`sudo apt install ffmpeg`)

## Installation
- If not done already: clone the [`sofa_env`](https://github.com/ScheiklP/sofa_env) repository and follow the instructionsof that repository to setup a conda environment and compile SOFA with SofaPython3 support. Do not forget to install the package itself with `pip install -e .` afterwards. Make sure that you did all the steps inside the `sofa` conda environment. If you have already installed SOFA with SofaPython3 in a conda env, it should be enough to pip install the repository.

- Clone this repository ([sofa_zoo](https://github.com/ScheiklP/sofa_zoo)).

- Make sure that the `sofa` conda environment is active.

- Install this repository with `pip install -e .`.

## Citing
If you use the project in your work, please consider citing it with:
```bibtex
@article{JMLR:v24:23-0207,
  author  = {Paul Maria Scheikl and Balázs Gyenes and Rayan Younis and Christoph Haas and Gerhard Neumann and Martin Wagner and Franziska Mathis-Ullrich},
  title   = {LapGym - An Open Source Framework for Reinforcement Learning in Robot-Assisted Laparoscopic Surgery},
  journal = {Journal of Machine Learning Research},
  year    = {2023},
  volume  = {24},
  number  = {368},
  pages   = {1--42},
  url     = {http://jmlr.org/papers/v24/23-0207.html}
}
```

## Acknowledgements
This work is supported by the Helmholtz Association under the joint research school "HIDSS4Health – Helmholtz Information and Data Science School for Health".
