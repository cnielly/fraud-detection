# Fraud Detection

## Introduction

This school project was completed by 5 students of the join degree *Data Science for Business*, between [Ecole polytechnique](https://www.polytechnique.edu/en) and [HEC Paris](https://www.hec.edu/en). 
It was done for a course titled "Born2Code", in partnership with [Quinten](https://www.quinten-france.com/) (Data Science Consulting firm). 

Fraud detection is an increasing center of focus since machine learning democratization. It has applications in various fields such as:
ecommerce, platform, social networks, insurances or banks. The implied costs are always consequent and the
patterns difficult to detect, even harder to automatically detect…

One of the main challenges in this specific problematic is to deal with highly imbalanced
datasets: imagine the number of credit card transaction per day, the large part of them is obviously not
fraudulent, which infringes us to reach a balanced dataset, as in other ML problematics. This case study is
centered on existing methods to tackle this problem.

## I. Business approach of the problem

The goal of this project is to help a bank to detect fraudulent credit card transactions. We think this is a major issue for banks for four main reasons:
1.  **The loss of money:** credit card replacement, refund, administrative costs... Banks are loosing huge amounts of money each year because of undetected frauds. In our dataset, we estimated this loss at $ 11 milions per year (= fraud amount / 2 * 365).
2.  **The customer experience:** having your money stolen sucks, and this is why banks need to detect frauds as soon as possible, before the transaction is confirmed. Customers are expecting safety and reactivity on that topic. 
3.  **Justice:** as part of the economical and the societal environment of a country, we think that banks are willing to fight against fraud and theft.It is important for bank to stay aligned with their values. 
4.  **Political credibility:** banks' reputation with regulation authorities is at stake here. Who wants to trust a bank whose customers are being stolen?

A suspicious activity can easily be detected by a human being. 
For instance, if Paul's typical payment is done in Paris, between 8am and 10pm, with an average amount of 50€, we can say that a transaction of 1000€ made at 3am from Madrid is suspicious...
As you can see, various informations can be used to assess the veracity of a transaction. Location, hour of the day and amount are examples of them. 
The challenge of a bank that deals with thousands of transactions each day is to automatize this classification process.

The goal of this project is to build an fully automated pipeline that indicated whereas a transaction is a fraud or not. 
We designed this pipeline avoiding to major pitfalls:
1.  **A fraud flies beneath the radar.** The fraud is not detected and the transaction order is accepted. The credit card needs to be
replaced, the client reassured and (eventually) refunded
2.  **A normal transaction is labeled as fraud by mistake.** The client credit card is blocked on an unfounded suspicion of fraud. The client cannot use his/her card properly. The credit card needs to be reactivated, the client reassured. 

To put it simple, we need to avoid **False Negatives** and **False Positives**. 

## II. Read the data

We worked on the famous Kaggle Dataset *Credit Card Fraud Detection*, that can be found [here](https://www.kaggle.com/mlg-ulb/creditcardfraud).<br/>
This dataset is extremely imbalanced: only 0.17% of transactions are frauds. 

## III. Our data science approach

The majority of the code can be found in the `/src/fraud/nodes` folder.
Each python file corresponds to a step of our process:
- Preprocess the data (`preprocessing.py` file): the columns that were not obtained thanks to the PCA are standardized. Then we separate the X dataset (30 columns) of the y dataset (target column).
- Find metrics to evaluate a pipeline performance (`metrics.py` file): in classification, there is a lot of metrics. Accuracy, Precision, Recall, Specificity, Geometric Mean, F1-Sore... They all have their pros and cons, and their explanability. We decided to focus on the average_precision_score to select our model. 
- Find the best pipeline (`modeling.py` file): in this file you will find all the functions needed to test several pipelines and compare them. Data modification (undersampling, oversampling), algorithm modification (thresholds, weights) and fine-tuning (grid search on hyperparameters).

After testing more than 650 combinations, we opted for the following pipeline:
- Partial undersampling using One Sided Selection (research paper [here]( https://sci2s.ugr.es/keel/pdf/algorithm/congreso/kubat97addressing.pdf)).
- Partial oversampling using SMOTE.
- LightGBM Classifier.

## Conclusion

With a greedy algorithm (Logistic Regression on raw data):

$$Precision = \frac{TP}{TP+FP} = 0.87$$.<br/>

$$Recall = \frac{TP}{TP+FN} = 0.64$$

With our pipeline:

$$Precision = \frac{TP}{TP+FP} = 0.87$$.<br/>

$$Recall = \frac{TP}{TP+FN} = 0.85$$

**To go further**<br/>
To reach higher scores, one can design a custom loss
$$Precision = \frac{TP}{TP+FP} = 0.87$$.<br/>

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
