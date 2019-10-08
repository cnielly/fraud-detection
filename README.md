# Fraud Detection

## Introduction

## Business approach of the problem

## Read the data

## Our data science approach

## Kedro

### Overview

This is your new Kedro project, which was generated using `Kedro 0.15.1` by running:

```
kedro new
```

Take a look at the [documentation](https://kedro.readthedocs.io) to get started.

### Rules and guidelines

In order to get the best out of the template:
 * Please don't remove any lines from the `.gitignore` file provided
 * Make sure your results can be reproduced by adding necessary data to `data/01_raw` only
 * Don't commit any data to your repository
 * Don't commit any credentials or local configuration to your repository
 * Keep all credentials or local configuration in `conf/local/`

### Installing dependencies

Dependencies have been be declared in `src/requirements.txt` for pip installation

To install them, run:

```
kedro install
```

### Running Kedro

You can run your Kedro project with:

```
kedro run
```


### Working with Kedro from notebooks

In order to use notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

For using Jupyter Lab, you need to install it:

```
pip install jupyterlab
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

You can also start Jupyter Lab:

```
kedro jupyter lab
```

And if you want to run an IPython session:

```
kedro ipython
```
