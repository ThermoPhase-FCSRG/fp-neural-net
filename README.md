# Biofuels flash point modelling using neural networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15112682.svg)](https://doi.org/10.5281/zenodo.15112682)

In this project, we have developed a neural network model capable of predicting the flash point of biofuel mixtures using simple features, such as the average molar mass of the mixture, the natural logarithm of the average vapor pressure, and the experimental method.


## Supporting material

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13633431.svg)](https://doi.org/10.5281/zenodo.13633431)

The experimental flash point data for 1-butanol and Fatty Acid Ethyl Esters, used in training dataset, is available under the Zenodo license above.

## Installing dependencies

Before installing the required dependencies, it's important to create a Python virtual environment, preferably using Python version 3.10:

```shell
python -m venv .venv
```
or (depending on your OS)
```shell
python3 -m venv .venv
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
pip install -e "."
```

To install the optional dependencies as well, run:
```shell
pip install -e ".[dev]"
```

## Usage

In the ```notebooks``` directory, there is a single notebook where the entire case study for the paper has been completed. Each step followed during the work can be reproduced.
Note that you should install the Python Notebook to your virtual env.

## Contact

Please feel free to contact us or open an issue/discussion if you have interesting in something, question, etc. You can also send an email to m241948@dac.unicamp.br if more convenient for you.
