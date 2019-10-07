import pandas as pd
import sklearn as sk
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score,confusion_matrix
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score

import random
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

def compute_metrics(y_test, y_pred, average='weighted', return_index=False):
    """
    Function computing metrics of interest for a sets of prediction

    :input y_test: pd.DataFrame or np.array of original label
    :input y_pred: pd.DataFrame or np.array of predicted label

    :output red: list of value for metrics, in order - Accuracy - Precision - Recall - F1 Score - Sensitivity - Specifity
    """
    if return_index:
        return ['accuracy', 'precision', 'recall', 'f1_score', 'sensitivity_score', 'specificity_score', 'geometric_mean_score', 'average_precision_score']
    else :
    	res = []
    	res.append(accuracy_score(y_test, y_pred))
    	res.append(precision_score(y_test, y_pred, average=average))
    	res.append(recall_score(y_test, y_pred, average=average))
    	res.append(f1_score(y_test, y_pred, average=average))
    	res.append(sensitivity_score(y_test, y_pred, average=average))
    	res.append(specificity_score(y_test, y_pred, average=average))
    	res.append(geometric_mean_score(y_test, y_pred, average=average))
    	res.append(average_precision_score(y_test, y_pred, average=average))
    	return res

def compute_main_metrics(y_test, y_pred, average='weighted', return_index=False):
    """
    Function computing metrics of interest for a sets of prediction

    :input y_test: pd.DataFrame or np.array of original label
    :input y_pred: pd.DataFrame or np.array of predicted label

    :output red: list of value for metrics, in order - Accuracy - Precision - Recall - F1 Score - Sensitivity - Specifity
    """
    if return_index:
        return ['precision', 'recall', 'average_precision_score']
    else :
    	res = []
    	res.append(precision_score(y_test, y_pred, average=average))
    	res.append(recall_score(y_test, y_pred, average=average))
    	res.append(average_precision_score(y_test, y_pred, average=average))
    	return res


def plot_confusion_matrix(y_true,y_pred, classes=['Not Fraud','Fraud'],
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix in heatmap.

    :input y_true: pd.DataFrame or np.array of original label
    :input y_pred: pd.DataFrame or np.array of predicted label
    :input title: title of the plot
    :input cmap: color of the heatmap

    :output: nice plot of confusion matrix heatmap
    """
    cm = sk.metrics.confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    tick_marks_vertical = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks_vertical, classes)


    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xlim([-0.5,1.5])
    plt.ylim([-0.5,1.5])




def tuning_sample_grid(X_train, y_train, X_test, y_test, model, grid):
    """
    Function computing the average precision score for a model and a set of parameters specified as a list of dict in grid. 
    
    :input X_train, y_train, X_test, y_test: pd.dataframe of np.array with eponym data
    :input model: model from SKLearn or follwoing SKLearn API
    :input grid: dict with parameters name as key and value as value

    :output scores: list cointainning scoress for each methods
    """
    scores = []
    #Loop to compute accuracy of each model
    print(len(grid))
    for i in range(len(grid)):
        grid_tested = grid[i]
        print(grid_tested)
        model.set_params(**grid_tested)
        model.fit(X=X_train, y=y_train)
        y_preds = model.predict(X_test)
        score = average_precision_score(y_test, y_preds)
        scores.append(score)
    return scores


def score_grid_search(X, y, model, grid, k, folds, method1, method2):
    """
    Function comparing metrics across different parameters on the average precision score using a k-Fold cross validation 

    :input X,y: pd.DataFrame or np.Array
    :input model: model from SKLearn or follwoing SKLearn API
    :input k: number of parameters to try
    :input folds: number of folds
    :input methods1, method2: oversampling/undersampling methods from ImbLearn or following similar API

    :output final: list cointainning scoress for each methods
    """
    kf = KFold(n_splits=folds)
    #select a subset of models
    Param_grid = random.choices(grid, k=k)
    
    #print(len(Param_grid))
    scores = []
    for train_index, test_index in kf.split(X):
        print('testing ')
        X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        
        X_train, y_train = method1.fit_resample(X=X_train, y=y_train)
        X_train, y_train = method2.fit_resample(X=X_train, y=y_train)
        
        scores.append(tuning_sample_grid(X_train=X_train, 
                           y_train=y_train, 
                           X_test=X_test, 
                           y_test=y_test, 
                           model=model,  
                           grid=Param_grid))

    final = pd.concat([pd.DataFrame(Param_grid), pd.DataFrame.from_records(scores).transpose().mean(axis=1)], axis = 1)
    return final
