from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def compute_metrics(y_test, y_pred):
	"""
	Function computing metrics of interest for a sets of prediction

	:input y_test: pd.DataFrame or np.array of original label
	:input y_pred: pd.DataFrame or np.array of predicted label

	:output red: list of value for metrics, in order - Accuracy - Precision - Recall - F1 Score
	"""
    res = []
    res.append(accuracy_score(y_test, y_pred))
    res.append(precision_score(y_test, y_pred, average='weighted'))
    res.append(recall_score(y_test, y_pred, average='weighted'))
    res.append(f1_score(y_test, y_pred,average='weighted'))
    return res