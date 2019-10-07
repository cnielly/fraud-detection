import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score,confusion_matrix
from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score

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




def tuning_sample_grid(X_train, y_train, X_test, y_test, model, nb_model, grid):

    best_score = 0

    #select a subset of models
    Param_grid = random.choices(grid, k = nb_model)

    #Loop to compute accuracy of each model
    for i in range(nb_model):
        grid_tested = Param_grid[i]
        print(grid_tested)
        model.set_params(**grid_tested)
        model.fit(X=X_train, y=y_train)
        y_preds = model.predict(X_test)
        score = average_precision_score(y_test, y_preds)
        if score > best_score:
            best_score = score
            best_grid = grid_tested

    print('Best average_precision_score is: ' + str(best_score))
    return best_grid
