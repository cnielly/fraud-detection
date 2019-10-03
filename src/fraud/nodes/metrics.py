
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score


def compute_metrics(y_test, y_pred, average='weighted'):
    """
    Function computing metrics of interest for a sets of prediction

    :input y_test: pd.DataFrame or np.array of original label
    :input y_pred: pd.DataFrame or np.array of predicted label

    :output red: list of value for metrics, in order - Accuracy - Precision - Recall - F1 Score - Sensitivity - Specifity
    """
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
	
def compute_main_metrics(y_test, y_pred, average='weighted'):
    """
    Function computing metrics of interest for a sets of prediction

    :input y_test: pd.DataFrame or np.array of original label
    :input y_pred: pd.DataFrame or np.array of predicted label

    :output red: list of value for metrics, in order - Accuracy - Precision - Recall - F1 Score - Sensitivity - Specifity
    """
    res = []
    res.append(precision_score(y_test, y_pred, average=average))
    res.append(recall_score(y_test, y_pred, average=average))
    res.append(average_precision_score(y_test, y_pred, average=average))

    return res