# Fraud Detection

## Introduction

This school project was completed by 5 students of the join degree Data Science for Business, between Ecole polytechnique and HEC Paris.
It was done for a course titled "Born2Code", in partnership with [Quinten](https://www.quinten-france.com/) (Data Science Consulting firm). 

Fraud detection is an increasing center of focus since machine learning democratization. It has applications in various fields such as:
ecommerce, platform, social networks, insurances or banks. The implied costs are always consequent and the
patterns difficult to detect, even harder to automatically detect…

One of the main challenges in this specific problematic is to deal with highly imbalanced
datasets: imagine the number of credit card transaction per day, the large part of them is obviously not
fraudulent, which infringes us to reach a balanced dataset, as in other ML problematics. This case study is
centered on existing methods to tackle this problem.

## I. Business approach of the problem

## II. Read the data

## III. Our data science approach

## Conclusion

## Bonus: Kedro

### A. Overview

This is your new Kedro project, which was generated using `Kedro 0.15.1` by running:

```
kedro new
```

Take a look at the [documentation](https://kedro.readthedocs.io) to get started.

### B. Rules and guidelines

In order to get the best out of the template:
 * Please don't remove any lines from the `.gitignore` file provided
 * Make sure your results can be reproduced by adding necessary data to `data/01_raw` only
 * Don't commit any data to your repository
 * Don't commit any credentials or local configuration to your repository
 * Keep all credentials or local configuration in `conf/local/`

### C. Installing dependencies

Dependencies have been be declared in `src/requirements.txt` for pip installation

To install them, run:

```
kedro install
```

### D. Running Kedro

You can run your Kedro project with:

```
kedro run
```


### E. Working with Kedro from notebooks

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
