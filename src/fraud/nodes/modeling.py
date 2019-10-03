from imblearn.metrics import sensitivity_score, specificity_score, geometric_mean_score 
from imblearn.under_sampling import RandomUnderSampler, NearMiss, EditedNearestNeighbours, NeighbourhoodCleaningRule
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.fraud.nodes.metrics import *



def sampling(X_train, y_train, X_test, y_test, sampling_instances, model_instances):
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