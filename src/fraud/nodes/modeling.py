from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score 
from imblearn.under_sampling import RandomUnderSampler, NearMiss, EditedNearestNeighbours, NeighbourhoodCleaningRule
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.fraud.nodes.metrics import *



def sampling(X_train, y_train, X_test, y_test, sampling_instances, model_instances):
    """
    Function to test different samlping methids on different models. 
    For each sampling methods in list(sampling_instance) and each model in list(models) compute the set of metrics of metrics function 

    :input X_train, y_train, X_test, y_test: np.array or pd.DataFrame of data
    :input sampling_instance: list of instances of sampling methods (tested for all methods of Imblearn)
    :input model_instances: list of instances of models (from SKLearn or with SKLearn API)

    :output metrics: nested list of metrics, with order Sampling1- Method1, Sampling1_model2, sampling1_model3....

    """



    metrics = []
    # go through all sampling methods
    for sampling_instance in sampling_instances:
        print('fitting sampling'+ str(sampling_instances.index(sampling_instance)))
        X_train, y_train = sampling_instance.fit_resample(X=X_train, y=y_train)
        
        # Go through all models
        for model_instance in model_instances:
            print('fitting model' + str(model_instances.index(model_instance)))
            model_instance.fit(X_train, y_train)
            metrics.append(compute_main_metrics(y_test, model_instance.predict(X_test)))

    return metrics

#e.g. : 
"""
LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
SVM = svm.LinearSVC()
RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

NCL = NeighbourhoodCleaningRule()
NM = NearMiss()
ENN = EditedNearestNeighbours()

models_instance = [LR, RF]
undersamplings = [NCL, NM, ENN]

output = sampling(X_train, y_train, X_test, y_test undersamplings, models_instance)

models = ['LogReg', 'SVM', 'RF']
methods = ['SM', 'ADA', 'KMSM', 'RO', 'SVMSM']
index = [model + '_' + method for model in models for method in methods]

output = pd.DataFrame(output, columns=['accuracy', 'precision', 'recall', 'f1_score', 'sensitivity_score', 'specificity_score', 'geometric_mean_score', 'average_precision_score'], index=index)
"""