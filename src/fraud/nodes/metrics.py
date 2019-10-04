import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
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


def plot_confusion_matrix(y_true,y_pred, classes,
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
