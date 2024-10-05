# Biofuels flash point modelling using neural networks

In this project, I've developed a neural network model capable of predicting the flash point of biofuel mixtures using simple features, such as the average molar mass of the mixture, the natural logarithm of the average vapor pressure, and the experimental method.


## Supporting material

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13633431.svg)](https://doi.org/10.5281/zenodo.13633431)

The experimental flash point data for 1-butanol and Fatty Acid Ethyl Esters, used in training dataset, is available under the Zenodo license above.

## Installing dependencies

Before installing the required dependencies, it's important to create a Python virtual environment, preferably using Python version 3.10.

```shell
python -m venv .venv
```

After creating the virtual environment, activate it:
- On Windows:
    ```shell
    .venv\Scripts\activate
    ```
- On macOS/Linux:
    ```shell
    source .venv/bin/activate
    ```
Now, install the required dependencies:
```shell
pip install .
```

To install the optional dependencies as well, run:
```shell
pip install .[dev]
```

## Usage

In the ```notebooks``` directory, there is a single notebook where I have completed an entire case study for my paper. You can reproduce each step that I followed during my work.

## Contact

Please feel free to contact me if you have interesting in something, question, etc.