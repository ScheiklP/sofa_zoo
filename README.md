# SOFA_ZOO
This repository is part of the project "LapGym - An Open Source Framework for Reinforcement Learning in Robot-Assisted Laparoscopic Surgery".
It provides the code for the reinforcement learning experiments as described in the [LapGym paper](https://arxiv.org/abs/2302.09606) for the environments of [sofa_env](https://github.com/ScheiklP/sofa_env).

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
@article{scheiklLapGym2023,
    authors = {Scheikl, Paul Maria and Gyenes, Balázs and Younis, Rayan and Haas, Christoph and Neumann, Gerhard and Mathis-Ullrich, Franziska and Wagner, Martin},
    title = {LapGym - An Open Source Framework for Reinforcement Learning in Robot-Assisted Laparoscopic Surgery},
    year = {2023},
    journal={arXiv preprint arXiv:2302.09606},
}
```

## Acknowledgements
This work is supported by the Helmholtz Association under the joint research school "HIDSS4Health – Helmholtz Information and Data Science School for Health".
